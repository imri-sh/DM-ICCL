import random

import numpy as np
import torch
from dataset_admin import ArcDataset, EmotionDataset
from model_loader import (
    get_flan_T5_base,
    get_flan_T5_large,
    get_flan_T5_xl,
    get_gemma_2_9b_instruct,
    get_llama_3_8b_instruct,
    get_phi2,
    get_phi3,
    get_phi3_5,
)


def get_args(parser):
    # Adding arguments to the parser
    parser.add_argument(
        "--dataset",
        type=str,
        default="arc",
        choices=["arc", "emotions"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--portions",
        type=float,
        nargs=3,
        help="List of 3 floats between 0 and 1",
    )

    parser.add_argument(
        "--kshot", type=int, default=2, help="Number of shots to inject"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="phi3_5",
        choices=[
            "phi2",
            "phi3",
            "phi3_5",
            "flan_t5_base",
            "flan_t5_large",
            "flan_t5_xl",
            "llama3_8b_instruct",
            "gemma2_9b_instruct",
        ],
        help="Name of model to use",
    )
    parser.add_argument(
        "--seed", type=str, default=42, help="Seed value for random number generator"
    )
    parser.add_argument(
        "--example_selector_type",
        type=str,
        default="random",
        help="The type of example selector to use [sim, random]",
    )
    parser.add_argument(
        "--encoder_path",
        type=str,
        default="all-MiniLM-L6-v2",
        help="The path of the encoder to use",
    )
    return parser


def prase_dataset_arg(dataset_arg):
    if dataset_arg == "arc":
        dataset = ArcDataset()
    elif dataset_arg == "emotions":
        dataset = EmotionDataset()
    else:
        raise ValueError(f"Invalid dataset arg: {dataset_arg}")

    return dataset


def parse_model_arg(model_arg):
    if model_arg == "phi2":
        model, tokenizer, model_name = get_phi2()
    elif model_arg == "phi3":
        model, tokenizer, model_name = get_phi3()
    elif model_arg == "phi3_5":
        model, tokenizer, model_name = get_phi3_5()
    elif model_arg == "flan_t5_base":
        model, tokenizer, model_name = get_flan_T5_base()
    elif model_arg == "flan_t5_large":
        model, tokenizer, model_name = get_flan_T5_large()
    elif model_arg == "flan_t5_xl":
        model, tokenizer, model_name = get_flan_T5_xl()
    elif model_arg == "llama3_8b_instruct":
        model, tokenizer, model_name = get_llama_3_8b_instruct()
    elif model_arg == "gemma2_9b_instruct":
        model, tokenizer, model_name = get_gemma_2_9b_instruct()
    else:
        raise ValueError(f"Invalid model arg: {model_arg}")
    return model, tokenizer, model_name


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
