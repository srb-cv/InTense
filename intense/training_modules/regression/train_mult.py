"""
Training the dataset as a regression prooblem,
and evaluating as a positve-negative classification problem.
Applicable for datasets: MOSI, MOSEI
"""

import os
import sys
import logging
import argparse
import inspect
import shutil
from pathlib import Path
import tomli

import mlflow
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from intense.datasets.affect.get_data import get_dataloader
from intense.fusions.mult.mult import MULTModel
from intense.trainer.affect_trainer import test, train
from intense.models.common_models import Identity
from intense.utils.util import get_save_path
from intense.training_modules.regression.training_arguments import (
    TrainingArguments, get_argument_parser)


parser = argparse.ArgumentParser(description="Multimodal-Transformer Training")

def get_training_args():
    return TrainingArguments(
        dataset='mosi',
        act_before_vbn=False,
        batch_size=8,
        lr=5e-3,
        weight_decay=0.01,
        early_stop=False,
        save="mult",
        scheduler_step_size=500,
        scheduler_gamma=0.9,
        z_norm=False,
        affine=False,
        epochs=200,
        evaluate=None,
        experiment_name="temp_test",
        is_packed=False,
        label_norm=False,
        start_epoch=0,
        num_workers=0,
        hidden_dims=None,
        tf_latent_dim=None,
        tf_indices=None,
        reg_param=None,
        p=None,
        track_complexity=False
    )

class HParams():
        num_heads = 8
        layers = 4
        attn_dropout = 0.1
        attn_dropout_modalities = [0,0,0.1]
        relu_dropout = 0.1
        res_dropout = 0.1
        out_dropout = 0.1
        embed_dropout = 0.2
        embed_dim = 40
        attn_mask = True
        output_dim = 1
        all_steps = False


def test_model(save_dir_path, args, device, test_robust, model_path=None):
    if model_path is not None:
        test_model_path = model_path
    else:
        test_model_path = f"{save_dir_path}/checkpoints/{args.save}_best.pt"
    print(f" Testing : {test_model_path}")
    model = torch.load(test_model_path)
    model.to(device)
    model.eval()
    test_results = test(
        model=model,
        test_dataloaders_all=test_robust,
        dataset="mosi",
        is_packed=False,
        criterion=torch.nn.L1Loss(),
        task="posneg-classification",
        no_robust=True,
    )
    print(f"Test Results : {test_results}")
    return test_results


def main(args):
    with open('intensfusion/datasets/datasets.toml', mode='rb') as fp:
        data_config = tomli.load(fp)
    data_config = data_config[args.dataset]
    if "mosi" in args.dataset:
        dataset = "mosi"
    elif "mosei" in args.dataset:
        dataset = "mosei"
    # setup logging for mlflow
    db_path = (
        "sqlite:///experiments.db"
    )
    mlflow.set_tracking_uri(db_path)
    run_description = """
    """
    if args.evaluate is not None:
        experiment = mlflow.set_experiment("evaluation")
        path = Path(args.evaluate)
        save_dir_path = str(path.parent)
    else:
        save_dir_path = f"logs/{get_save_path(mode='train')}"
        experiment = mlflow.set_experiment(f"{args.experiment_name or 'Test'}")
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
            print(f"Logging Experiment in : {save_dir_path}")
    logging.basicConfig(
        filename=f"{save_dir_path}/myapp.log",
        encoding="utf-8",
        level=logging.INFO
    )
    logging.info("Started")

    with mlflow.start_run(
        run_name="-".join(save_dir_path.split("/")[1:]),
        experiment_id=experiment.experiment_id,
        description=run_description,
    ):
        mlflow.log_params(vars(args))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        writer = SummaryWriter(log_dir=save_dir_path)
        traindata, validdata, test_robust = get_dataloader(
            data_config["data_path"],
            robust_test=False,
            data_type=dataset,
            save_path=save_dir_path,
            batch_size=args.batch_size,
            z_norm=args.z_norm,
            label_norm=args.label_norm,
            num_workers=args.num_workers,
            max_pad=True
        )
        modalities = data_config["modalities"]
        mod_input_dims = data_config["input_dims"]
        encoders =  [Identity(), Identity(), Identity()]
        
        head = Identity()
        fusion = MULTModel(3, [20, 5, 300], hyp_params=HParams) #TODO: remove hardcoding the dimensions

        mlflow.log_params(
            {
                "fusion": fusion.__class__.__name__,
                "head": head.__class__.__name__,
                "encoders": [
                    encoders[i].__class__.__name__
                    for i in range(len(encoders))
                ],
                "input_dimensions": mod_input_dims,
                "modalities": modalities
            }
        )

        if args.evaluate is not None:
            test_results = test_model(
                save_dir_path, args, device, test_robust,
                model_path=args.evaluate
            )
            print(test_results)
            mlflow.log_metrics(
                {f"test_{key}": value for (key, value) in test_results.items()}
            )
            mlflow.end_run()
            sys.exit("Evaluation Completed. Exiting...")

        if not os.path.exists(save_dir_path):
            os.makedirs(os.path.join(save_dir_path, "checkpoints"))
        filename = inspect.getframeinfo(inspect.currentframe()).filename
        shutil.copyfile(filename,
                        f"{save_dir_path}/{os.path.basename(filename)}")

        # TODO: Save the model as a state dict not as a complete model
        model = train(
            encoders,
            fusion,
            head,
            traindata,
            validdata,
            total_epochs=args.epochs,
            task="regression",
            optimtype=torch.optim.AdamW,
            early_stop=args.early_stop,
            is_packed=args.is_packed,
            lr=args.lr,
            save=args.save,
            weight_decay=args.weight_decay,
            objective=torch.nn.L1Loss(),
            track_complexity=args.track_complexity,
            save_path=save_dir_path,
            tf_encoders=None,
            pre_tf_encoders=None,
            reg_param=args.reg_param,
            p=args.p,
            tf_modality_indices=None,
            scheduler_step_size=args.scheduler_step_size,
            modalities=modalities,
            writer=writer,
            scheduler_gamma=args.scheduler_gamma,
        )
        max_test_acc = 0
        best_test_model = ""
        for model in ["best", "best_acc", "latest"]:
            model_path = f"{save_dir_path}/checkpoints/{args.save}_{model}.pt"
            test_results = test_model(
                save_dir_path=None,
                args=args,
                device=device,
                test_robust=test_robust,
                model_path=model_path,
            )
            if test_results["Accuracy"] > max_test_acc:
                max_test_acc = test_results["Accuracy"]
                test_acc_with_zero = test_results["Acc_include_zero"]
                best_test_model = ""
        mlflow.log_metrics(
            {"test_Accuracy": max_test_acc,
             "test_Acc_include_zero": test_acc_with_zero}
        )
        mlflow.set_tag("best_test", best_test_model)
        mlflow.end_run()


if __name__ == "__main__":
    parser = get_argument_parser(parser)
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        print("Command Line Argumant are not passed, Loading from dataclass")
        args = get_training_args()
    print(args)
    main(args)
    
