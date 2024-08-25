import argparse
from datetime import datetime
from pathlib import Path

import utils
from args_utils import get_args, prase_dataset_arg, set_seed
from experiments import Experiments
from model_loader import set_dtype, ModelLoader
from utils import plots_dir, data_mapping_jsons_dir
import torch
from data_mapping import data_mapping


def run_experiments(args):
    set_dtype(fp_type="fp16")
    experiments = Experiments(args)
    print(experiments)
    experiments.experiment_acc_over_k(
        ks=args.kshots,
        title=f"Model: {args.model} \n Dataset: {args.dataset}",
    )


def test_data_mapping(args):
    set_seed(args.seed)
    set_dtype(fp_type="fp16")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, model_name = ModelLoader.get_model_and_tokenizer(args.model, device=device)
    dataset = utils.trim_data(prase_dataset_arg(args.dataset), args.portions)
    ks = [1,2]
    mean_confidences = []
    for k in ks :
        num_evals = args.num_evals if k != 0 else 1
        plot_path = plots_dir / f"{model_name}_{dataset.get_name()}_k_{k}_num_evals_{num_evals}.png"
        plot_title = f"{model_name}, {dataset.get_name()} Data Map"
        results, mean_confidence = data_mapping(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            num_evals=num_evals,
            k_shots=k,
            title= plot_title,
            plot_path=plot_path,
            show_plot=True
        )
        data_mapping_jsons_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        save_path = data_mapping_jsons_dir / f"{model_name}_{dataset.get_name()}_k_{k}_num_evals_{num_evals}_{timestamp}.json"
        utils.save_results(results, save_path=save_path)
        mean_confidences.append(mean_confidence)

    for i, k in enumerate(ks):
        print(f"k={k}, mean confidence={mean_confidences[i]*100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Run the experiment")
    args = get_args(parser).parse_args()
    # if args.datamap:
    #     print("Performing data mapping...")
    #     test_data_mapping(args)
    # else:
    #     print("Running evaluation experiments...")
    #     # args.portions = (0.1, 0.1, 0.1)
    #     # args.model = "llama3_8b_instruct"
    #     run_experiments(args)

    # for model in [
    #     "phi3_5",
    #     "flan_t5_xl",
    #     "llama3_8b_instruct",
    #     "gemma2_9b_instruct",
    # ]:
    #     args.model = model
    for model in ["llama3_8b_instruct", "gemma2_9b_instruct", "phi3_5"]:
        args.model = model
        args.dataset = "agnews"
        args.kshots = [0, 1, 3, 10]
        args.portions = (0.1, 0.01, 0.01)
        run_experiments(args)

if __name__ == '__main__':
    main()
