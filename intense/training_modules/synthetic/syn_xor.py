import argparse
import inspect
import logging
import os
import shutil
import sys
import tomli
from math import log
from pathlib import Path
from zlib import Z_NO_COMPRESSION

import mlflow
import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter


from intense.datasets.synthetic.get_data import MutltiModalCharDataset, get_dataloader
from intense.fusions.mkl.mkl_fusion import MKLFusionVectorwiseBatchNorm
from intense.trainer.affect_trainer import test, train
from intense.models.common_models import Identity
from intense.models.char_model import CharModelBatchNorm
from intense.utils.functional import _build_pre_tf_models,\
    _build_tensor_fusion_models
from intense.utils.util import get_save_path
from intense.training_modules.synthetic.training_arguments\
    import TrainingArguments, get_argument_parser

parser = argparse.ArgumentParser(description="PyTorch Training for Synthetic"
                                 "Data")

def get_training_args():
    return TrainingArguments(
        tf_latent_dim=4,
        dataset='synthgene_xor',
        act_before_vbn=False,
        batch_size=64,
        lr=5e-3,
        weight_decay=0.01,
        early_stop=False,
        save="sarcasm_intense",
        reg_param=0.1,
        p=1,
        scheduler_step_size=500,
        scheduler_gamma=0.9,
        z_norm=False,
        affine=False,
        epochs=20,
        evaluate=None,
        experiment_name="syn_xor_intense_2",
        hidden_dims=[32, 32, 32],
        is_packed=True,
        label_norm=False,
        start_epoch=0,
        num_workers=4,
        tf_indices=['12','13','23','123']
    )

def test_model(save_dir_path, args, test_robust, model_path=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        dataset=args.dataset,
        is_packed=True,
        criterion=nn.CrossEntropyLoss(),
        task="classification",
        no_robust=True,
    )
    print(f"Test Results : {test_results}")
    return test_results



def main(args):
    # setup logging
    tags = {
        "Dataset": f"{args.dataset} Data"
    }
    with open('intense/datasets/datasets.toml', mode='rb') as fp:
        data_config = tomli.load(fp)
    data_config = data_config[args.dataset]
    db_path = (
        "sqlite:///logs/experiments.db"
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
    mlflow.start_run(
        run_name="-".join(save_dir_path.split("/")[1:]),
        experiment_id=experiment.experiment_id,
        description=run_description,
    )
    mlflow.log_params(vars(args))
    mlflow.set_tags(tags)
    writer = SummaryWriter(log_dir=save_dir_path)
    
    dataset: MutltiModalCharDataset = MutltiModalCharDataset(
            data_csv_path=data_config['data_path']
        )
    traindata, validdata, test_robust = get_dataloader(
            dataset=dataset, batch_size=args.batch_size
        )
    num_classes = data_config["num_classes"]
    modalities = data_config["modalities"]
    # mod_input_dims = data_config["input_dims"]
    mod_hidden_dims = args.hidden_dims
    encoders = [CharModelBatchNorm(output_dim=dim) for dim in args.hidden_dims]
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

    fusion = MKLFusionVectorwiseBatchNorm(
        in_features_dict=in_features_dict,
        out_features=num_classes,
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
            # "input_dimensions": mod_input_dims,
            "feature_dimensions": mod_hidden_dims,
            "modalities": modalities,
            "tf_modality_indices": tf_modality_indices,
        }
    )

    if args.evaluate is not None:
        test_results = test_model(
            save_dir_path, args, test_robust,
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
        task="classification",
        optimtype=torch.optim.AdamW,
        early_stop=args.early_stop,
        is_packed=args.is_packed,
        lr=args.lr,
        save=args.save,
        weight_decay=args.weight_decay,
        objective=nn.CrossEntropyLoss(),
        save_path=save_dir_path,
        tf_encoders=tf_encoders,
        pre_tf_encoders=pre_tf_encoders,
        reg_param=args.reg_param,
        p=args.p,
        tf_modality_indices=tf_modality_indices,
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
            test_robust=test_robust,
            model_path=model_path,
        )
        if test_results["Accuracy"] > max_test_acc:
            max_test_acc = test_results["Accuracy"]
            best_test_model = ""
    mlflow.log_metrics(
        {"test_Accuracy": max_test_acc}
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