import random
import numpy as np
import torch
from dataset_admin import AgNewsDataset, ArcDataset

def get_args(parser):
    # Adding arguments to the parser
    parser.add_argument(
        "--datasets",
        type=str,
        default="arc",
        # choices=["arc", "emotions"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--portions",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="List of 3 floats between 0 and 1",
    )
    parser.add_argument(
        "--test_size", type=int, default=500, help="unified test set size"
    )
    parser.add_argument(
        "--train_size", type=int, default=500, help="unified train set size"
    )
    parser.add_argument(
        "--kshots", type=int, nargs="+", default=[0, 1], help="List of k shots to use"
    )

    # parser.add_argument(
    #     "--test_kshots_list", type=int, nargs=3, default=[0, 1,], help="List of k shots to use"
    # )

    parser.add_argument(
        "--datamap_kshots", type=int, nargs=1, default=3, help="kshot for data-mapping"
    )

    parser.add_argument("--datamap", type=bool, default=False)
    parser.add_argument(
        "--models",
        type=str,
        default="flan_t5_base,phi2",
        # choices=[
        #     "phi2",
        #     "phi3",
        #     "phi3_5",
        #     "flan_t5_base",
        #     "flan_t5_large",
        #     "flan_t5_xl",
        #     "llama3_8b_instruct",
        #     "gemma2_9b_instruct",
        # ],
        help="Name of model to use",
    )
    parser.add_argument("--num_evals", type=int, default=5, help="Number of evaluations for each example in datamap")
    parser.add_argument(
        "--seed", type=str, default=42, help="Seed value for random number generator"
    )
    parser.add_argument(
        "--example_selector_type",
        type=str,
        default="random",
        choices=["random", "similarity","datamap"],
        help="The type of example selector to use",
    )
    parser.add_argument(
        "--encoder_path",
        type=str,
        default="all-MiniLM-L6-v2",
        help="The path of the encoder to use in similarity based example selector",
    )
    return parser


def prase_dataset_arg(dataset_arg):
    if dataset_arg == "arc":
        dataset = ArcDataset()
    elif dataset_arg == "agnews":
        dataset = AgNewsDataset()
    else:
        raise ValueError(f"Dataset {dataset_arg} is not supported.")
    return dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
