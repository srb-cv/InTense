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

from intense.datasets.synthetic.get_data import (
    get_dataloader, MutltiModalCharDataset
)
from intense.fusions.mkl.mkl_fusion import (
    MKLFusionVectorwiseBatchNorm_V1,
)
from intense.trainer.affect_trainer import test, train
from intense.models.encoders_syn import CharModelBatchNorm
from intense.models.common_models import Identity
from intense.utils.util import get_save_path

parser = argparse.ArgumentParser(description="PyTorch Training for MOSEI")
parser.add_argument(
    "--tf-latent-dim",
    type=int,
    default=16,
    help="Latent dimension for the tensor fusion layer."
    "Every modality's representation is reduced to this dimension",
)
parser.add_argument(
    "--latent-dim",
    type=int,
    default=32,
    help="Latent dimension for the Encoder."
    "Every modality's representation is reduced to this dimension",
)
parser.add_argument(
    "--epochs", type=int, default=200,
    help="Number of epochs to train")
parser.add_argument(
    "--start-epoch", type=int, default=0,
    help="Starting epoch for resuming training"
)
parser.add_argument(
    "--batch-size", type=int, default=32,
    help="Batch size for training"
)
parser.add_argument(
    "--lr", type=float, default=1e-3,
    help="Learning rate for training")
parser.add_argument(
    "--weight-decay", type=float, default=0.01,
    help="Weight decay for training"
)
parser.add_argument(
    "--early-stop", type=bool, default=False,
    help="Early stopping for training"
)
parser.add_argument(
    "--save",
    type=str,
    default="mosei_mkl_baseline",
    help="Name of the model to be saved",
)
parser.add_argument(
    "--reg-param",
    type=float,
    default=0.01,
    help="Regularization parameter for the MKL fusion layer",
)
parser.add_argument(
    "--p",
    type=int, default=1,
    help="p value for the MKL fusion layer")
parser.add_argument(
    "--scheduler-step-size", type=int, default=500,
    help="Step size for the scheduler"
)
parser.add_argument(
    "--scheduler-gamma",
    type=float,
    default=0.9,
    help="Scheduler gamma factor for the optimizer",
)
parser.add_argument(
    "--evaluate",
    type=str,
    default=None,
    help="Path to the model to be evaluated"
)
parser.add_argument(
    "--act-before-norm",
    help="Relu activation before the vecotwise batch normalization",
    action="store_true",
)
parser.add_argument(
    "--affine",
    help='''Affine transformation by a scalar applied
    to the normalized feature representation''',
    action="store_true",
)
parser.add_argument(
    "--experiment-name",
    type=str,
    default=None,
    help="logging the experiment name for mlflow",
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
                        no_robust=True)
    print(f"Test Results : {test_results}")
    return test_results


if __name__ == "__main__":
    args = parser.parse_args()
    mlflow.set_tracking_uri(
        "sqlite:///experiments.db"
    )
    tags = {
        "Dataset": "Synth Gene",
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
        filename=f"{save_dir_path}/myapp.log", encoding="utf-8", level=logging.INFO
    )

    logging.info("Started")
    run_description = """
    """
    with mlflow.start_run(
        run_name="-".join(save_dir_path.split("/")[1:]),
        experiment_id=experiment.experiment_id,
        description=run_description,
    ):
        mlflow.log_params(vars(args))
        mlflow.set_tags(tags)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        writer = SummaryWriter(log_dir=save_dir_path)
        dataset: MutltiModalCharDataset = MutltiModalCharDataset(
            data_csv_path="data/synthetic_data/synth_gene.csv"
        )
        traindata, validdata, test_robust = get_dataloader(
            dataset=dataset, batch_size=args.batch_size,
            num_workers=2
        )
        modalities = dataset.modalities
        num_classes = 2
        logging.info(f"modalities:{modalities}")
        enc_out_dims = [args.latent_dim for _ in range(len(modalities))]
        encoders = [CharModelBatchNorm(output_dim=dim) for dim in enc_out_dims]

        # tf_modality_indices = ['13','23','12','123']
        # tf_latent_dim = args.tf_latent_dim

        head = Identity()
        in_features_dict = {
            index: num_features
            for (index, num_features) in zip(modalities, enc_out_dims)
        }
        fusion = MKLFusionVectorwiseBatchNorm_V1(
            in_features_dict=in_features_dict,
            out_features=2,
            affine=args.affine,
            activation=args.act_before_norm,
        )

        mlflow.log_params(
            {
                "fusion": fusion.__class__.__name__,
                "head": head.__class__.__name__,
                "encoders": [
                    encoders[i].__class__.__name__ for i in range(len(encoders))
                ],
                "feature_dimensions": enc_out_dims,
                "modalities": modalities,
            }
        )

        if args.evaluate is not None:
            test_results = test_model(
                save_dir_path, args, device, test_robust, model_path=args.evaluate
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
        model = train(
            encoders,
            fusion,
            head,
            traindata,
            validdata,
            total_epochs=args.epochs,
            optimtype=torch.optim.AdamW,
            early_stop=args.early_stop,
            lr=args.lr,
            save=args.save,
            weight_decay=args.weight_decay,
            track_complexity=False,
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
            data_type="synthetic"
        )
                
        model_path = f"{save_dir_path}/checkpoints/{args.save}_best.pt"
        test_results = test_model(
            save_dir_path=None,
            args=args,
            device=device,
            test_robust=test_robust,
            model_path=model_path,
        )        
        test_acc = test_results["Accuracy"]
        test_acc_with_zero = test_results["Acc_include_zero"]
        mlflow.log_metrics(
            {"test_Accuracy": test_acc}
        )
        mlflow.log_metrics({"test_Accuracy": test_acc})
        mlflow.end_run()
