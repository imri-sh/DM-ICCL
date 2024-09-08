import argparse
import pandas as pd

from datetime import datetime
from itertools import product
from preprocess import preprocess_datamaps
from utils import *
from args_utils import get_args
from experiments import Experiments
from model_loader import set_dtype


def run_experiments(args, timestamp: str = "", show_plot: bool = True, save_results_bool: bool = True):
    set_dtype(fp_type="fp16")
    experiments = Experiments(args)
    # example_selectors_types = args.example_selectors_types

    example_selectors_types_datamap = ["datamap", "datamap_similarity"]
    experiment_results_path = (
            experiment_results_dir
            / f"experiment_results_{timestamp}.csv"
    )
    experiment_results = pd.DataFrame()
    experiment_results = run_experiments_base(experiments, experiment_results, experiment_results_path, timestamp)
    run_experiments_datamap_eval(experiments, experiment_results, experiment_results_path, timestamp)


def run_experiments_datamap_eval(experiments: Experiments, experiment_results: pd.DataFrame,
                                 experiment_results_path: Path, timestamp: str = "", save_results_bool: bool = True):
    args = experiments.args
    example_selectors_types = ["datamap", "datamap_similarity"]
    num_of_experiments = len(args.models) * len(args.datasets) * len(example_selectors_types) * len(args.orders)
    print(f"Total number of experiments for datamap_eval: {num_of_experiments}")

    for i, (model_name, dataset_name, selector, order) in enumerate(
            product(args.models, args.datasets, example_selectors_types, args.orders)):
        print(f"Running experiment {i + 1}/{num_of_experiments}, \n"
              f"models {model_name}, using selector {selector}, with order {order}")


def run_experiments_base(experiments: Experiments, experiment_results: pd.DataFrame, experiment_results_path: Path,
                         timestamp: str = "", show_plot: bool = False, save_results_bool: bool = True):
    args = experiments.args
    # example_selectors_types = args.example_selectors_types
    example_selectors_types = ["random", "similarity"]
    num_of_experiments = len(args.models) * len(args.datasets) * len(example_selectors_types)
    plot_data = []

    print(f"Total number of experiments for baseline: {num_of_experiments}")

    # experiment_results = pd.DataFrame()

    for i, (model_name, dataset_name, selector) in enumerate(
            product(args.models, args.datasets, example_selectors_types)):

        print(f"Running experiment {i + 1}/{num_of_experiments}, using selector {selector}")
        experiments.reset_seed()
        experiments.set_model(model_name=model_name)
        experiments.set_dataset(dataset_name=dataset_name, portions=args.portions, sizes=args.sizes)
        args.example_selector_type = selector

        accs = experiments.experiment_acc_over_k(
            title=f"Model: {model_name} \n Dataset: {dataset_name}",
            show_plot=False,
            timestamp=timestamp
        )

        col_name = f"{dataset_name}_{model_name}"
        for k, acc in accs.items():
            row_name = f"{selector}_{k}"
            experiment_results.loc[row_name, col_name] = acc
            if save_results_bool:
                experiment_results.to_csv(experiment_results_path, index=True)
            if show_plot:
                plot_data.append({
                    'kshots': k,
                    'accuracy': acc,
                    'model_dataset': f"{model_name}_{dataset_name}",
                    'example_selector_type': selector
                })

    print(experiment_results)

    if show_plot:
        plot_code_exiled(plot_data, timestamp)

    return experiment_results


def main():
    parser = argparse.ArgumentParser(description="Run the experiment")
    args = get_args(parser).parse_args()
    # args.models = "llama3_8b_instruct,llama_3_8B"
    # args.models = "llama3_8b_instruct,phi3_5,"
    # args.datasets = "agnews,arc"
    args.datasets = "arc"
    # args.example_selectors_types = ["random", "similarity"]
    args.kshots = [0, 1, 2]
    # args.kshots_datamap_similarity = [[1, 2, 3], [3, 2, 1]]
    # args.kshots_datamap_similarity = [[1, 2, 3], [3, 2, 1], [5, 1, 0], [4, 2, 0], [2, 4, 0], [0, 4, 2],
    #                                   [0, 2, 4], [2, 0, 4], [4, 0, 2], [6, 0, 0], [0, 6, 0], [0, 0, 6], ]
    args.datamap_kshots = 3
    args.num_evals = 5
    # args.models = args.models.split(',')
    args.models = [
        # "llama3_8b_instruct",
        # "llama_3_8b",
        # "phi3_5",
        # "phi2",
        'flan_t5_base',
        'flan_t5_large',
        # "gemma2_9b_instruct",
        # "gemma2_9b",
    ]
    args.datasets = args.datasets.split(',')
    # args.sizes = None
    args.portions = None
    args.sizes = [1119, 299, 1172]
    # args.sizes = [1119, 50, 1172]
    # args.sizes = [50, 15, 15]
    print("########################## Stage 1: Datamap Constructions ########################################")
    timing_info = preprocess_datamaps(models=args.models,
                                      datasets=args.datasets,
                                      portions=args.portions,
                                      sizes=args.sizes,
                                      datamap_kshots=args.datamap_kshots,
                                      num_evals=args.num_evals,
                                      seed=args.seed,
                                      save_and_show_plots=True)
    print(timing_info)
    print("####################################### DONE ##################################################")

    print("########################## Stage 2: Experiments ###############################################")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    experiment_results_path = run_experiments_base(args, timestamp)
    # experiment_results = load_results(experiment_results_path)
    print("####################################### DONE ##################################################")
    print(experiment_results_path)


if __name__ == '__main__':
    main()
