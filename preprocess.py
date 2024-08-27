from itertools import product
from pathlib import Path

from safetensors import torch

import utils
from args_utils import set_seed, prase_dataset_arg
from data_mapping import data_mapping
from experiments import Experiments
from model_loader import set_dtype, ModelLoader
import torch

import time
import pandas as pd

import time
import pandas as pd


def preprocess_datamaps(models, datasets, portions=None, sizes=None, datamap_kshots=None, num_evals: int = 5,
                        seed: int = 42, save_and_show_plots=True):
    set_dtype(fp_type="fp16")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed=seed)

    num_of_datamaps = len(models) * len(datasets)
    pp_datamaps_dir, pp_datamaps_results_dir, pp_datamaps_plots_dir = utils.get_datamaps_dir_paths()
    # List to accumulate timing information
    timing_data = []

    # Track total time for the function

    total_start_time = time.time()

    for i, (model_name, dataset_name) in enumerate(product(models, datasets)):
        model, tokenizer, model_name = ModelLoader.get_model_and_tokenizer(model_name, device=device)
        dataset = utils.trim_data(prase_dataset_arg(dataset_name), portions, sizes)
        train_set, _, _ = dataset.get_data()
        train_size = len(train_set)
        datamap_path = pp_datamaps_results_dir / f"dm_{model_name}_{dataset_name}_train_size_{train_size}_k_{datamap_kshots}_num_evals_{num_evals}.json"

        if not datamap_path.exists():
            print(f"Creating datamap {i + 1} out of {num_of_datamaps} datamaps.\n")
            print("---------------------------------------------------------------")
            print(f"\t Model Name: {model_name}\n"
                  f"\t Dataset Name: {dataset_name}\n"
                  f"\t Number of Evaluations per example: {num_evals}\n"
                  f"\t trained with k_shots: {datamap_kshots}\n"
                  f"\t train size: {train_size}\n"
                  f"\t device: {device}\n")
            print("---------------------------------------------------------------")

            # Start time for this datamap
            datamap_start_time = time.time()

            datamapping_results, _ = data_mapping(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                num_evals=num_evals,
                k_shots=datamap_kshots,
                title=f"{model_name}, {dataset_name} Data Map",
                plot_path=pp_datamaps_plots_dir /  f"dm_{model_name}_{dataset_name}_train_size_{train_size}_k_{datamap_kshots}_num_evals_{num_evals}.png",
                show_plot=save_and_show_plots
            )

            # End time for this datamap
            datamap_end_time = time.time()
            datamap_duration = datamap_end_time - datamap_start_time

            print(f"Datamap created successfully. Saving in {datamap_path}")
            print(f"Time taken for this datamap: {datamap_duration:.2f} seconds")
            print("---------------------------------------------------------------")

            # Save results and timing information
            utils.save_results(datamapping_results, save_path=datamap_path)
            timing_data.append({
                'Model': model_name,
                'Dataset': dataset_name,
                'Train Size': train_size,
                'Datamap Creation Time (s)': datamap_duration
            })

        else:
            print("---------------------------------------------------------------------------------")
            print(f"Datamap already exists in {datamap_path}. Skipping...")
            print("---------------------------------------------------------------------------------")

    # End time for the entire function
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    print("---------------------------------------------------------------")
    print(f"Total time taken for all datamaps: {total_duration:.2f} seconds")
    print("---------------------------------------------------------------")

    # Create a DataFrame from timing data
    timing_df = pd.DataFrame(timing_data)

    # Save the timing DataFrame as a CSV or JSON file
    timing_df_path = pp_datamaps_dir / "datamaps_timing_info.csv"
    timing_df.to_csv(timing_df_path, index=False)

    # Optionally save as JSON
    # timing_df_path_json = pp_datamaps_results_dir / "datamap_timing_info.json"
    # timing_df.to_json(timing_df_path_json, orient='records', lines=True)

    print(f"Timing information saved to {timing_df_path}")

    return timing_df


# Example usage:

