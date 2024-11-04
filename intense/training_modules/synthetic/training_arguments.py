from dataclasses import dataclass, fields

@dataclass
class TrainingArguments:
    """
    Class for writing arguments for training the model


    Parameters:
        test_dir:
            Test directory for arguments
        test_lr:
            Test learning rate for arguments
    """
    latent_dim: int
    lr: float
    epochs: int
    start_epoch: int
    batch_size: int
    weight_decay: float
    early_stop: bool
    save: str
    scheduler_step_size: int
    scheduler_gamma: float
    evaluate: bool
    experiment_name: str
    mod_index: int


def classFromArgs(className, argDict):
    fieldSet = {f.name for f in fields(className) if f.init}
    filteredArgDict = {k: v for k, v in argDict.items() if k in fieldSet}
    return className(**filteredArgDict)


def get_argument_parser(parser):
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=32,
        help="Latent dimension for the Encoder."
        "Every modality's representation is reduced to this dimension",
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Number of epochsto train")
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
        "--experiment-name",
        type=str,
        default=None,
        help="logging the experiment name for mlflow",
    )
    parser.add_argument(
        "--mod-index",
        type=int,
        default=0,
        help="Index of the modality in the multimodal dataset"
        "on which the modael is trained"
    )
    return parser
