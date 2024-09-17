from itertools import product
import utils
from args_utils import set_seed, prase_dataset_arg
from data_mapping import data_mapping
from model_loader import set_dtype, ModelLoader
import torch


def preprocess_datamaps(models, datasets, sizes=None, datamap_kshots=None, num_evals: int = 5,
                        seed: int = 42, save_and_show_plots=True):

    set_dtype(fp_type="fp16")
    set_seed(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_of_datamaps = len(models) * len(datasets)
    pp_datamaps_dir, pp_datamaps_results_dir, pp_datamaps_plots_dir = utils.get_datamaps_dir_paths()
    for i, (model_name, dataset_name) in enumerate(product(models, datasets)):
        dataset = utils.trim_data(prase_dataset_arg(dataset_name),sizes)
        train_set, _, _ = dataset.get_data()
        train_size = len(train_set)
        datamap_path = pp_datamaps_results_dir / f"dm_{model_name}_{dataset_name}_train_size_{train_size}_k_{datamap_kshots}_num_evals_{num_evals}.json"

        if not datamap_path.exists():
            model, tokenizer, model_name = ModelLoader.get_model_and_tokenizer(
                model_name, device=device
            )
            print(f"Creating datamap {i + 1} out of {num_of_datamaps} datamaps.\n")
            print("---------------------------------------------------------------")
            print(f"\t Model Name: {model_name}\n"
                  f"\t Dataset Name: {dataset_name}\n"
                  f"\t Number of Evaluations per example: {num_evals}\n"
                  f"\t trained with k_shots: {datamap_kshots}\n"
                  f"\t train size: {train_size}\n"
                  f"\t device: {device}\n")
            print("---------------------------------------------------------------")
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
            print(f"Datamap created successfully. Saving in {datamap_path}")
            print("---------------------------------------------------------------")
            utils.save_results(datamapping_results, save_path=datamap_path)

        else:
            print("---------------------------------------------------------------------------------")
            print(f"Datamap already exists in {datamap_path}. Skipping...")
            print("---------------------------------------------------------------------------------")