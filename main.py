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
    example_selectors_types = args.example_selectors_types
    num_of_experiments = len(args.models) * len(args.datasets)
    print(f"Total number of experiments: {num_of_experiments * len(example_selectors_types)}")

    plot_data = []

    for i, selector in enumerate(example_selectors_types):
        print(
            f"-- Starting Evaluation experiments with {selector} example selector {(i + 1)}/{len(example_selectors_types)} --")
        args.example_selector_type = selector
        experiment_results = pd.DataFrame()

        for j, (model_name, dataset_name) in enumerate(product(args.models, args.datasets)):
            print(f"Running experiment {j + 1}/{num_of_experiments}")
            experiments.reset_seed()
            experiments.set_model(model_name=model_name)
            experiments.set_dataset(dataset_name=dataset_name, portions=args.portions, sizes=args.sizes)

            accs = experiments.experiment_acc_over_k(
                title=f"Model: {model_name} \n Dataset: {dataset_name}",
                show_plot=False,
                timestamp=timestamp
            )

            for k, acc in accs.items():
                col_name = f"{dataset_name}_{model_name}"
                experiment_results.loc[k, col_name] = acc
                plot_data.append({
                    'kshots': k,
                    'accuracy': acc,
                    'model_dataset': f"{model_name}_{dataset_name}",
                    'example_selector_type': selector
                })

        print(experiment_results)

        experiment_results_path = (
                experiment_results_dir
                / f"experiment_results_{args.example_selector_type}_{timestamp}.csv"
        )
        if save_results_bool:
            experiment_results.to_csv(experiment_results_path, index=True)

    plot_df = pd.DataFrame(plot_data)

    if show_plot:
        # Create subplots for each model-dataset pair
        unique_datasets_models = plot_df['model_dataset'].unique()
        num_subplots = len(unique_datasets_models)
        fig, axes = plt.subplots(nrows=num_subplots, figsize=(14, 5 * num_subplots))

        for ax, dataset_model in zip(axes, unique_datasets_models):
            subset_df = plot_df[plot_df['model_dataset'] == dataset_model]

            sns.lineplot(
                data=subset_df,
                x='kshots',
                y='accuracy',
                hue='example_selector_type',
                style='example_selector_type',
                markers=True,
                dashes=False,
                linewidth=2.5,
                ax=ax
            )

            ax.set_title(f'Results for {dataset_model}', fontsize=16)
            ax.set_xticks(np.arange(min(subset_df['kshots']), max(subset_df['kshots']) + 1, 1))
            ax.set_xlabel('kshots', fontsize=14)
            ax.set_ylabel('Accuracy', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(title='Example Selector Type', fontsize=12, title_fontsize=14)

        plt.tight_layout()
        plot_save_path = plots_dir / f'experiment_results_{timestamp}_subplots.png'
        plt.savefig(plot_save_path, bbox_inches='tight')
        plt.show()

    return experiment_results_path


def main():
    parser = argparse.ArgumentParser(description="Run the experiment")
    args = get_args(parser).parse_args()
    # args.models = "llama3_8b_instruct,llama_3_8B"
    # args.models = "llama3_8b_instruct,phi3_5,"
    # args.datasets = "agnews,arc"
    args.datasets = "arc"
    # args.example_selectors_types = ["random", "similarity", "datamap", "datamap_similarity"]
    args.example_selectors_types = ["random", "similarity"]
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
        'flan_t5_large'
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
    experiment_results_path = run_experiments(args, timestamp)
    # experiment_results = load_results(experiment_results_path)
    print("####################################### DONE ##################################################")
    print(experiment_results_path)


if __name__ == '__main__':
    main()
