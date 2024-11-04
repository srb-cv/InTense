'''
Tested for training on following classification datasets:
1. MOSEI(sentiment), 2. MOSI(sentiment)
'''
import argparse
import inspect
import logging
import os
import shutil
import sys
from pathlib import Path

import mlflow
import tomli
import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter

from intense.datasets.affect.get_data import get_dataloader
from intense.fusions.mkl.mkl_fusion import MKLFusionVectorwiseBatchNorm
from intense.models.common_models import GRUwithBatchNorm, Identity
from intense.trainer.affect_trainer import test, train
from intense.training_modules.regression.training_arguments import (
    TrainingArguments, get_argument_parser)
from intense.utils.functional import (_build_pre_tf_models,
                                           _build_tensor_fusion_models)
from intense.utils.util import get_save_path

parser = argparse.ArgumentParser(description="Training module for InTense")

def get_training_args():
    return TrainingArguments(
        tf_latent_dim=8,
        dataset='mosi',
        act_before_vbn=False,
        batch_size=16,
        lr=5e-3,
        weight_decay=0.01,
        early_stop=False,
        save="sarcasm_intense",
        reg_param=0.05,
        p=1,
        scheduler_step_size=500,
        scheduler_gamma=0.9,
        z_norm=False,
        affine=False,
        epochs=200,
        evaluate=None,
        experiment_name="mosi_intens",
        hidden_dims=[32, 32, 64],
        is_packed=True,
        label_norm=False,
        start_epoch=0,
        num_workers=4,
        tf_indices=['13']
    )


def test_model(save_dir_path, args, device, test_robust,
               test_task, objective, is_packed, model_path=None):
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
        is_packed=is_packed,
        criterion=objective,
        task=test_task,
        no_robust=True,
    )
    print(f"Test Results : {test_results}")
    return test_results

def main(args):
    with open('intense/datasets/datasets.toml', mode='rb') as fp:
        data_config = tomli.load(fp)
    data_config = data_config[args.dataset]
    if "mosi" in args.dataset:
        dataset = "mosi"
    elif "mosei" in args.dataset:
        dataset = "mosei"
    else:
        dataset = args.dataset
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
    mlflow.start_run(
        run_name="-".join(save_dir_path.split("/")[1:]),
        experiment_id=experiment.experiment_id,
        description=run_description,
    )
    mlflow.log_params(vars(args))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=save_dir_path)
    
    modalities = data_config["modalities"]
    mod_input_dims = data_config["input_dims"]
    train_task = data_config["train_task"]
    test_task = data_config["test_task"]
    mod_hidden_dims = args.hidden_dims
    traindata, validdata, test_robust = get_dataloader(
        data_config["data_path"],
        robust_test=False,
        data_type=dataset,
        save_path=save_dir_path,
        batch_size=args.batch_size,
        z_norm=args.z_norm,
        label_norm=args.label_norm,
        num_workers=args.num_workers,
        task=train_task
    )
    
    encoders = [
        GRUwithBatchNorm(
            indim, hiddim, dropout=True, has_padding=True, batch_first=True
        )
        for (indim, hiddim) in zip(mod_input_dims, mod_hidden_dims)
    ]

    modality_indices = [i + 1 for i in range(len(modalities))]
    tf_modality_indices = args.tf_indices
    tf_latent_dim = args.tf_latent_dim

    pre_tf_encoders: dict[str, nn.Module] = _build_pre_tf_models(
        tf_modality_indices,
        latent_dim_dict={str(i+1): mod_hidden_dims[i]
                            for i in range(len(mod_hidden_dims))},
        out_dim=tf_latent_dim,
    )
    tf_encoders: list[nn.Module] = _build_tensor_fusion_models(
        tf_modalities=tf_modality_indices, input_dim=tf_latent_dim
    )
    head = Identity()
    in_features = mod_hidden_dims + [
        tf_latent_dim ** len(tf_modality_indices[i])
        for i in range(len(tf_modality_indices))
    ]
    all_modality_indices = modality_indices + tf_modality_indices
    in_features_dict = {
        str(index): num_features
        for (index, num_features) in zip(all_modality_indices, in_features)
    }
    print(f"The in features to the fusion module are: {in_features}")
    out_features = 1 if train_task=='regression' else data_config["num_classes"]
    fusion = MKLFusionVectorwiseBatchNorm(
        in_features_dict=in_features_dict,
        out_features=out_features,
        affine=args.affine,
        activation=args.act_before_vbn,
    )

    mlflow.log_params(
        {
            "fusion": fusion.__class__.__name__,
            "head": head.__class__.__name__,
            "encoders": [
                encoders[i].__class__.__name__
                for i in range(len(encoders))
            ],
            "input_dimensions": mod_input_dims,
            "feature_dimensions": mod_hidden_dims,
            "modalities": modalities,
            "tf_modality_indices": tf_modality_indices,
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
    if train_task == "regression":
        objective = torch.nn.L1Loss()
    elif train_task == "classification":
        objective = torch.nn.CrossEntropyLoss()
    else:
        raise Exception("Training Objective can only be regression or classification")
    # TODO: Save the model as a state dict not as a complete model
    model = train(
        encoders,
        fusion,
        head,
        traindata,
        validdata,
        total_epochs=args.epochs,
        task=train_task,
        optimtype=torch.optim.AdamW,
        early_stop=args.early_stop,
        is_packed=args.is_packed,
        lr=args.lr,
        save=args.save,
        weight_decay=args.weight_decay,
        objective=objective,
        track_complexity=args.track_complexity,
        save_path=save_dir_path,
        tf_encoders=tf_encoders,
        pre_tf_encoders=pre_tf_encoders,
        reg_param=args.reg_param,
        p=args.p,
        tf_modality_indices=tf_modality_indices,
        scheduler_step_size=args.scheduler_step_size,
        writer=writer,
        scheduler_gamma=args.scheduler_gamma,
    )
    
    model_path = f"{save_dir_path}/checkpoints/{args.save}_best.pt"
    test_results = test_model(
            save_dir_path=None,
            args=args,
            device=device,
            test_robust=test_robust,
            test_task=test_task,
            objective=objective,
            is_packed=args.is_packed,
            model_path=model_path
        )
    try:
        test_acc_with_zero = test_results["Acc_include_zero"]
    except KeyError:
        test_acc_with_zero = 0.0
    mlflow.log_metrics(
        {"test_Accuracy": test_results["Accuracy"],
         "test_Acc_include_zero": test_acc_with_zero}
    )

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
    

