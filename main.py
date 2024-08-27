import argparse
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd

import utils
from preprocess import preprocess_datamaps
from utils import *
from args_utils import get_args, prase_dataset_arg, set_seed
from experiments import Experiments
from model_loader import set_dtype, ModelLoader
from utils import plots_dir, data_mapping_jsons_dir, save_results
import torch
from data_mapping import data_mapping


def run_experiments(args, timestamp: str = "", show_plot:bool=True, save_results_bool:bool=True):
    set_dtype(fp_type="fp16")
    experiments = Experiments(args)
    experiment_results_path = experiment_results_dir/ f"experiment_results_{args.example_selector_type}_{timestamp}.csv"
    example_selectors_types = ["datamap", "random", "similarity"]
    num_of_experiments = len(args.models) * len(args.datasets)
    print(f"Total number of experiments: {num_of_experiments * len(example_selectors_types)}")
    for i, selector in enumerate(example_selectors_types):
        print(f"-- Starting Evaluation experiments with  {selector} example selector {(i+1)}/{len(example_selectors_types)} --")
        # if selector != "datamap":
        #     args.kshots = [0, 3]
        # else:
        #     args.kshots = [0, 1]
        args.example_selector_type = selector
        experiment_results = pd.DataFrame()
        # Insert a 'kshots' column at the first position of the DataFrame and set it as the index.
        experiment_results.insert(0, 'kshots', args.kshots)
        experiment_results.set_index('kshots', inplace=True)
        # Loop through each combination of models and datasets.
        for j, (model_name, dataset_name) in enumerate(product(args.models, args.datasets)):
            # Print the progress of the experiment.
            print(f"Running experiment {j + 1}/{num_of_experiments}")
            # Set the model for the current experiment.
            experiments.set_model(model_name=model_name)
            # Set the dataset and the portion of data to be used for the current experiment.
            experiments.set_dataset(dataset_name=dataset_name,
                                    portions=args.portions,
                                    sizes=args.sizes)
            # print(experiments)

            # Collect accuracy results over different k-shot configurations.
            accs = experiments.experiment_acc_over_k(
                title=f"Model: {model_name} \n Dataset: {dataset_name}",
                show_plot=False,
                timestamp=timestamp
            )

            # Populate the DataFrame with accuracy values for each k-shot configuration.
            for k, acc in accs.items():
                col_name = f"{dataset_name}_{model_name}"
                experiment_results.loc[k, col_name] = acc

            # Print the current state of the experiment results DataFrame.
            print(experiment_results)

        # Save the results to a CSV file if `save_results_bool` is True.
        if save_results_bool:
            experiment_results.to_csv(experiment_results_path, index=True)

        # Generate and save plots if `show_plot` is True.
        if show_plot:
            plot_experiments(experiment_results_path, plot_path=plots_dir / f'experiment_results_{timestamp}.png')

    # Return the path to the saved experiment results.
    return experiment_results_path


def test_data_mapping(args):
    set_seed(args.seed)
    set_dtype(fp_type="fp16")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, model_name = ModelLoader.get_model_and_tokenizer(args.model, device=device)
    dataset = trim_data(prase_dataset_arg(args.dataset), args.portions)
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
        save_results(results, save_path=save_path)
        mean_confidences.append(mean_confidence)
    for i, k in enumerate(args.kshots):
        print(f"k={k}, mean confidence={mean_confidences[i]*100:.2f}%")



def main():
    parser = argparse.ArgumentParser(description="Run the experiment")
    args = get_args(parser).parse_args()
    # args.models = "llama3_8b_instruct,llama_3_8B"
    args.models = "flan_t5_base"
    args.datasets = "arc,agnews"
    args.kshots = [0, 1]
    args.models = args.models.split(',')
    args.datasets = args.datasets.split(',')
    args.portions = None
    # args.sizes = [1119, 299, 1172]
    args.sizes = [50, 15, 15]
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
    experiment_results = run_experiments(args, timestamp)
    print(experiment_results)
    print("####################################### DONE ##################################################")

if __name__ == '__main__':
    main()
