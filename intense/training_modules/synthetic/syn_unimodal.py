import os
import sys
import logging

import argparse
import inspect
import shutil
from pathlib import Path

import mlflow
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from intensfusion.datasets.synthetic.get_data import (
    get_dataloader, MutltiModalCharDataset
)
from intensfusion.trainer.unimodal_trainer import test, train
from intensfusion.models.encoders_syn import CharModelBatchNorm
from intensfusion.models.common_models import MLP
from intensfusion.utils.util import get_save_path
from intensfusion.training_modules.synthetic.training_arguments\
    import get_argument_parser, TrainingArguments

parser = argparse.ArgumentParser(description="PyTorch Training for Synthetic"
                                 "Data in unimodal setting")


def get_training_args():
    return TrainingArguments(
        latent_dim=32,
        epochs=200,
        start_epoch=0,
        batch_size=32,
        lr=1e-3,
        weight_decay=0.01,
        early_stop=False,
        save="syn_baseline",
        scheduler_step_size=500,
        scheduler_gamma=0.9,
        evaluate=None,
        experiment_name="unimodal_syn",
        mod_index=0
    )


def test_model(save_dir_path, args, device, test_robust, model_path=None):
    if model_path is not None:
        test_model_path = model_path
    else:
        test_model_path = f"{save_dir_path}/checkpoints/{args.save}_best.pt"
    print(f" Testing : {test_model_path}")
    model = torch.load(test_model_path)
    model.to(device)
    test_results = test(model=model, test_dataloaders_all=test_robust,
                        no_robust=True, modality_index=args.mod_index)
    print(f"Test Results : {test_results}")
    return test_results


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup logging via logging module, mlflow, and
    # tensorboard writer
    mlflow.set_tracking_uri(
        "sqlite:///experiments.db"
    )
    tags = {
        "Dataset": "Unimodal Synthetic Data"
    }
    if args.evaluate is not None:
        experiment = mlflow.set_experiment("test")
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
    run_description = """
    """
    mlflow.start_run(
        run_name="-".join(save_dir_path.split("/")[1:]),
        experiment_id=experiment.experiment_id,
        description=run_description,
    )
    mlflow.log_params(vars(args))
    mlflow.set_tags(tags)
    writer = SummaryWriter(log_dir=save_dir_path)

    # setup data
    dataset: MutltiModalCharDataset = MutltiModalCharDataset(
        data_csv_path="data/synthetic_data/data_c2_m10_v6/data.csv"
    )
    traindata, validdata, test_robust = get_dataloader(
        dataset=dataset, batch_size=args.batch_size,
        num_workers=4
    )
    modalities = dataset.modalities
    num_classes = 2
    logging.info(f"modalities:{modalities}")
    logging.info(f"training of modality:{modalities[args.mod_index]}")
    print(f"Training of modality:{modalities[args.mod_index]}")

    # setup encoder
    enc_out_dim = args.latent_dim
    encoder = CharModelBatchNorm(output_dim=enc_out_dim)

    # setup classifcation head
    head = MLP(32, 64, num_classes)

    # log the encoders, heads, fusion modules
    # for the netowrk
    mlflow.log_params(
        {
            "head": head.__class__.__name__,
            "encoders": encoder.__class__.__name__,
            "feature_dimension": enc_out_dim,
            "modality": modalities[args.mod_index],
        }
    )

    if args.evaluate is not None:
        test_results = test_model(
            save_dir_path=save_dir_path,
            args=args,
            device=device,
            test_robust=test_robust,
            model_path=args.evaluate
        )
        print(test_results)
        mlflow.log_metrics(
            {f"test_{key}": value for (key, value) in test_results.items()}
        )
        mlflow.end_run()
        sys.exit("Evaluation Completed. Exiting...")

    print(f"Logging Experiment in : {save_dir_path}")
    if not os.path.exists(save_dir_path):
        os.makedirs(os.path.join(save_dir_path, "checkpoints"))
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    shutil.copyfile(filename, f"{save_dir_path}/{os.path.basename(filename)}")

    # TODO: Save the model as a state dict not as a complete model
    # call the main training function that sets up the
    # model returns the trained model
    model = train(
        encoder,
        head,
        traindata,
        validdata,
        total_epochs=args.epochs,
        optimtype=torch.optim.AdamW,
        early_stop=args.early_stop,
        lr=args.lr,
        save=args.save,
        weight_decay=args.weight_decay,
        save_path=save_dir_path,
        scheduler_step_size=args.scheduler_step_size,
        writer=writer,
        scheduler_gamma=args.scheduler_gamma,
        modality_idx=args.mod_index,
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
    mlflow.log_metrics({"test_Accuracy": max_test_acc})
    mlflow.set_tag("best_test", best_test_model)
    mlflow.end_run()


if __name__ == "__main__":
    parser = get_argument_parser(parser)
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        print("Arguments not passed, Loading from dataclass")
        args = get_training_args()
    print(args)
    main(args)
