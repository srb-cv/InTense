from dataclasses import dataclass


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
    tf_latent_dim: int
    dataset: str
    lr: float
    epochs: int
    start_epoch: int
    batch_size: int
    weight_decay: float
    early_stop: bool
    save: str
    reg_param: float
    p: float
    scheduler_step_size: int
    scheduler_gamma: float
    evaluate: bool
    experiment_name: str
    z_norm: bool
    is_packed: bool
    label_norm: bool
    affine: bool
    act_before_vbn: bool
    hidden_dims: list
    num_workers: int
    tf_indices: list


def get_argument_parser(parser):
    parser.add_argument(
        "--tf-latent-dim",
        type=int,
        default=16,
        help="Latent dimension for the tensor fusion layer."
        "Every modality's representation is reduced to this dimension",
    )
    parser.add_argument(
        "--dataset", type=str, default='mosi',
        help="""dataset to be loaded for training, supports one of
        [mosi_raw, mosei_raw, mosi_senti, mosei_senti]"""
    )
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs to train")
    parser.add_argument(
        "--start-epoch", type=int, default=0,
        help="Starting epoch for resuming training"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for training")
    parser.add_argument(
        "--weight-decay", type=float, default=0.01,
        help="Weight decay for training"
    )
    parser.add_argument(
        "--early-stop", action="store_true",
        help="Early stopping for training"
    )
    parser.add_argument(
        "--save",
        type=str,
        default="mosi_mkl_baseline",
        help="Name of the model to be saved",
    )
    parser.add_argument(
        "--reg-param",
        type=float,
        default=0.01,
        help="Regularization parameter for the MKL fusion layer",
    )
    parser.add_argument("--p", type=int, default=1,
                        help="p value for the MKL fusion layer")
    parser.add_argument(
        "--scheduler-step-size", type=int, default=500,
        help="Step size for the scheduler"
    )
    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=0.1,
        help="Scheduler gamma factor for the optimizer",
    )
    parser.add_argument(
        "--z-norm", help="Z normalization for the dataset",
        action="store_true"
    )
    parser.add_argument(
        "--is-packed", help="Packed sequences for the dataset",
        action="store_true"
    )
    parser.add_argument(
        "--label-norm",
        help="Normalize the training labels to be in  range(-1,1)",
        action="store_true",
    )
    parser.add_argument(
        "--affine",
        help="""Affine transformation by a scalar applied to the normalized
        feature representation""",
        action="store_true",
    )
    parser.add_argument(
        "--act-before-vbn",
        help="Relu activation before the vecotwise batch normalization",
        action="store_true",
    )
    parser.add_argument(
        "--evaluate", type=str, default=None,
        help="Path to the model to be evaluated"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for logging in MLFlow",
    )
    parser.add_argument(
        "-hdims",
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[64, 32, 512],
        help="length of the encoder's output for each modality"
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Number of workers"
    )
    parser.add_argument(
        "--tf-indices",
        type=str,
        nargs="+",
        default=['12','13','23','123'],
        help="""the modality indices for which the interactions are
        considered.
        """
    )
    return parser
