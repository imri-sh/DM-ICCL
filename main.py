import argparse
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd

import utils
from args_utils import get_args, prase_dataset_arg, set_seed
from experiments import Experiments
from model_loader import set_dtype, ModelLoader
from utils import plots_dir, data_mapping_jsons_dir, save_results
import torch
from data_mapping import data_mapping

def run_experiments(args, show_plot=True):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    set_dtype(fp_type="fp16")
    experiments = Experiments(args)
    # Initialize an empty DataFrame with datasets and models as columns, and k-shots as rows
    experiment_results = pd.DataFrame()
    experiment_results.insert(0, 'kshots', args.kshots)
    experiment_results.set_index('kshots', inplace=True)

    num_of_experiments = len(args.models)*len(args.datasets)
    for i, (model_name, dataset_name) in enumerate(product(args.models, args.datasets)):
        print(f"Running experiment {i+1}/{num_of_experiments}")
        # print(experiments)
        experiments.set_model(model_name=model_name)
        experiments.set_dataset(dataset_name=dataset_name, portions=args.portions)

        # Collect accuracy results over k-shots
        accs = experiments.experiment_acc_over_k(
            ks=args.kshots,
            title=f"Model: {model_name} \n Dataset: {dataset_name}",
            show_plot=False,
            timestamp=timestamp
        )

        # Populate the DataFrame with accuracy values
        for k, acc in accs.items():
            col_name = f"{dataset_name}_{model_name}"
            experiment_results.loc[k, col_name] = acc
        print(experiment_results)

    experiment_results_dir = utils.results_dir / "experiments"
    experiment_results_dir.mkdir(exist_ok=True)
    experiment_results_path = experiment_results_dir/ f'experiment_results_{timestamp}.csv'
    experiment_results.to_csv(experiment_results_path, index=True)

    if show_plot:
        utils.plot_experiments(experiment_results_path, plot_path=plots_dir / f'experiment_results_{timestamp}.png')
    return experiment_results_path

def test_data_mapping(args):
    set_seed(args.seed)
    set_dtype(fp_type="fp16")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, model_name = ModelLoader.get_model_and_tokenizer(args.model, device=device)
    dataset = utils.trim_data(prase_dataset_arg(args.dataset), args.portions)
    # ks = [1,2]
    mean_confidences = []
    for k in args.kshots :
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

    for i, k in enumerate(args.kshots):
        print(f"k={k}, mean confidence={mean_confidences[i]*100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Run the experiment")
    args = get_args(parser).parse_args()
    args.models = args.models.split(',')
    args.datasets = args.datasets.split(',')
    # if args.datamap:
    #     print("Performing data mapping...")
    #     test_data_mapping(args)
    # else:
    print("Running evaluation experiments...")
    run_experiments(args)
    # df = pd.read_csv(utils.results_dir/ "experiments/experiment_results_20240825_1719.csv")
    # print(df)
    # df.rename(columns={'Unnamed: 0': 'kshots'}, inplace=True)
    # plot_experiments(df)

if __name__ == '__main__':
    main()
