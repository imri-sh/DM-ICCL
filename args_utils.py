import ast
import random
import numpy as np
import torch
from dataset_admin import AgNewsDataset, ArcDataset



def get_args(parser):
    parser.add_argument(
        "--datasets",
        type=str,
        default="arc,agnews",
        help="Dataset to use",
    )
    parser.add_argument('--sizes', type=int, nargs='+', required=True, help='sizes to use from train,val, test sets in this order')

    parser.add_argument(
        "--models",
        type=str,
        default="llama3_8b_instruct,llama_3_8b,phi3_5,gemma2_9b_instruct,gemma2_9b",
        help="Name of model to use",
    )


    parser.add_argument('--kshots', type=int, nargs='+', required=True, help='List of k-shot values for baselines experiments')

    parser.add_argument(
        "--orders", type=str, default="E-A-H,E-H-A,A-E-H,A-H-E,H-E-A,H-A-E",
        help="List of orders to use for the datamap selector"
    )

    parser.add_argument(
        "--datamap_kshots", type=int, nargs=1, default=3, help="kshot for data-mapping construction"
    )

    parser.add_argument("--num_evals",
                        type=int,
                        default=5,
                        help="Number of evaluations for each example in datamap")
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed value for random number generator"
    )
    parser.add_argument(
        "--eval_test_set",  action='store_true', help="True for evaluating on test set, if not specified using validation"
    )

    default_similarity = [[1, 2, 3], [3, 2, 1], [5, 1, 0], [4, 2, 0], [2, 4, 0], [0, 4, 2], [0, 2, 4], [2, 0, 4],
                          [4, 0, 2], [6, 0, 0], [0, 6, 0], [0, 0, 6]]

    parser.add_argument('--kshots_datamap_similarity', type=str, default=str(default_similarity), help='Nested list for k-shot datamap similarity')

    return parser


def prase_dataset_arg(dataset_arg):
    if dataset_arg == "arc":
        dataset = ArcDataset()
    elif dataset_arg == "agnews":
        dataset = AgNewsDataset()
    else:
        raise ValueError(f"Dataset {dataset_arg} is not supported.")
    return dataset

def parse_strings_to_lists(args):
    print(args.orders)
    args.orders = args.orders.split(',')
    args.models = args.models.split(',')
    args.datasets = args.datasets.split(',')
    if args.kshots_datamap_similarity:
        args.kshots_datamap_similarity = ast.literal_eval(args.kshots_datamap_similarity)
    return args
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
