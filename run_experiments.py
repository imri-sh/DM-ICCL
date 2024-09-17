import argparse
import pandas as pd

from datetime import datetime
from itertools import product

import args_utils
from preprocess_dm import preprocess_datamaps
from utils import *
from args_utils import get_args
from experiments import Experiments
from model_loader import set_dtype

def run_experiments(args, timestamp: str = "", show_plot: bool = False, save_results_bool: bool = True,run_baselines=True,run_datamaps=True, eval_test_set=True):
    set_dtype(fp_type="fp16")
    experiments = Experiments(args)

    experiment_results_path = (
            experiment_results_dir
            / f"experiment_results_{timestamp}.csv"
    )
    experiment_results = pd.DataFrame()
    if eval_test_set:
        print("Evaluating on test sets")
    else:
        print("Evaluating on validation set")

    if run_baselines:
        experiment_results = run_experiments_base(experiments,
                                                  experiment_results,
                                                  experiment_results_path,
                                                  eval_test_set,
                                                  timestamp,
                                                  show_plot,
                                                  save_results_bool)
    if run_datamaps:
        run_experiments_datamap_eval(experiments,
                                     experiment_results,
                                     experiment_results_path,
                                     eval_test_set,
                                     timestamp,
                                     save_results_bool)
    print(experiment_results_path)


def run_experiments_datamap_eval(experiments: Experiments, experiment_results: pd.DataFrame,
                                 experiment_results_path: Path, eval_test_set:bool =True,timestamp: str = "", save_results_bool: bool = True):
    args = experiments.args
    example_selectors_types = ["datamap", "datamap_similarity"]
    num_of_experiments = len(args.models) * len(args.datasets) * len(example_selectors_types) * len(args.orders)
    print(f"Total number of experiments for datamap_eval: {num_of_experiments}")

    for i, (model_name, dataset_name, selector, order) in enumerate(
            product(args.models, args.datasets, example_selectors_types, args.orders)):
        print(f"Running experiment {i + 1}/{num_of_experiments} with: \n"
              f"Model - {model_name}\n"
              f"Dataset - {dataset_name}\n"
              f"Selector - {selector}\n"
              f"Order - {order}\n")
        experiments.reset_seed()
        experiments.set_model(model_name=model_name)
        experiments.set_dataset(dataset_name=dataset_name, sizes=args.sizes)
        args.example_selector_type = selector

        accs = experiments.experiment_acc_over_k(
            ks=experiments.args.kshots_datamap_similarity,
            title=f"Model: {model_name} \n Dataset: {dataset_name}",
            show_plot=False,
            timestamp=timestamp,
            order=order,
            eval_test_set=eval_test_set
        )

        col_name = f"{dataset_name}_{model_name}"
        for k, acc in accs.items():
            row_name = f"{selector}_{k}_{order}"
            experiment_results.loc[row_name, col_name] = acc
            if save_results_bool:
                experiment_results.to_csv(experiment_results_path, index=True)

    print(experiment_results)

    return experiment_results


def run_experiments_base(experiments: Experiments, experiment_results: pd.DataFrame, experiment_results_path: Path,eval_test_set:bool=True,
                         timestamp: str = "", show_plot: bool = False, save_results_bool: bool = True):
    args = experiments.args
    example_selectors_types = ["random", "similarity"]
    num_of_experiments = len(args.models) * len(args.datasets) * len(example_selectors_types)
    plot_data = []

    print(f"Total number of experiments for baseline: {num_of_experiments}")

    for i, (model_name, dataset_name, selector) in enumerate(
            product(args.models, args.datasets, example_selectors_types)):

        print(f"Running experiment {i + 1}/{num_of_experiments} with: \n"
              f"Model - {model_name}\n"
              f"Dataset - {dataset_name}\n"
              f"Selector - {selector}\n")
        experiments.reset_seed()
        experiments.set_model(model_name=model_name)
        experiments.set_dataset(dataset_name=dataset_name, sizes=args.sizes)
        args.example_selector_type = selector

        accs = experiments.experiment_acc_over_k(
            ks=experiments.args.kshots,
            title=f"Model: {model_name} \n Dataset: {dataset_name}",
            show_plot=False,
            timestamp=timestamp,
            eval_test_set = eval_test_set
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
    # print(args)
    args_utils.parse_strings_to_lists(args)

    print("################ Stage 1: Datamap Constructions & Difficulty Assignment ##################")
    preprocess_datamaps(models=args.models,
                        datasets=args.datasets,
                        sizes=args.sizes,
                        datamap_kshots=args.datamap_kshots,
                        num_evals=args.num_evals,
                        seed=args.seed,
                        save_and_show_plots=True)
    print("####################################### DONE ##################################################")

    print("########################## Stage 2: Experiments ###############################################")
    run_experiments(args,
                    timestamp= datetime.now().strftime("%Y%m%d_%H%M"),
                    show_plot= False,
                    save_results_bool= True,
                    run_baselines=True,
                    run_datamaps=True,
                    eval_test_set=args.eval_test_set)
    print("####################################### DONE ##################################################")

if __name__ == '__main__':
    main()
