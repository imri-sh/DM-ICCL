from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import LogitsProcessor

import example_selectors
from utils import plots_dir, results_dir, datamap_plots_dir, data_mapping_jsons_dir
import utils
from model_loader import ModelLoader
from args_utils import  set_seed, prase_dataset_arg
from datetime import datetime
from example_selectors import ExampleSelectorFactory

class LimitTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_tokens):
        self.allowed_tokens = allowed_tokens

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, -float("inf"))
        mask[:, self.allowed_tokens] = 0
        return scores + mask

class Experiments:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = None
        self.model, self.tokenizer, self.model_name = None, None, None
        set_seed(args.seed)

    def set_model(self, model_name: str):
        self.model, self.tokenizer, self.model_name = ModelLoader.get_model_and_tokenizer(model_name, device=self.device)

    def set_dataset(self, dataset_name: str, portions=None, sizes=None):
        self.dataset_name = dataset_name
        self.dataset = utils.trim_data(prase_dataset_arg(dataset_name), portions, sizes)

    def experiment_acc_over_k(self, title: str="", show_plot:bool=True,timestamp:str=""):
        plot_path, results_path = self.generate_result_paths(timestamp)
        pp_datamaps_dir, pp_datamaps_results_dir, pp_datamaps_plots_dir = utils.get_datamaps_dir_paths()
        # datamap_path = data_mapping_jsons_dir / "flan-t5-base_ARC-challenge dataset_k_3_num_evals_5_20240826_1843.json"
        train_set, _, _ = self.dataset.get_data()
        datamap_results_path = pp_datamaps_results_dir / f"dm_{self.model_name}_{self.dataset_name}_train_size_{len(train_set)}_k_{self.args.datamap_kshots}_num_evals_{self.args.num_evals}.json"
        datamap_plot_path = pp_datamaps_plots_dir / f"dm_{self.model_name}_{self.dataset_name}_train_size_{len(train_set)}_k_{self.args.datamap_kshots}_num_evals_{self.args.num_evals}.png"
        example_selector_type = self.args.example_selector_type
        dataset_name = self.dataset.get_name()
        ks = self.args.kshots
        accs = {}
        kwargs = {
                 'model':self.model,
                 'tokenizer':self.tokenizer,
                 'model_name':self.model_name,
                 'dataset':self.dataset,
                 'examples': train_set,
                 'num_evals':self.args.num_evals,
                 'datamap_kshots': self.args.datamap_kshots,
                 'datamap_results_path': datamap_results_path,
                 'datamap_plot_path': datamap_plot_path,
                 'key':'question'}
        example_selector = ExampleSelectorFactory.get_example_selector(example_selector_type=example_selector_type,
                                                                       **kwargs)

        for k in tqdm(ks,desc=f"Evaluating model {self.model_name} on {dataset_name} with kshots {ks} using {example_selector_type} example selector"):
            accuracy, _, _ = self.evaluate_model(example_selector, k)
            accs[k] = accuracy
            print(f"kshot={k}, accuracy={accuracy * 100:.2f}% ")

        k_range = np.array(list(accs.keys())) * 3 if example_selector_type == 'datamap' else np.array(list(accs.keys()))
        if show_plot:
            utils.plot_accuracies_over_kshots(
                k_range=k_range,
                accuracies=list(accs.values()),
                title=title,
                filepath=plot_path,
            )
        utils.save_results(accs, save_path=results_path)
        return accs

    def generate_result_paths(self, timestamp:str=""):
        dataset_name = self.dataset.get_name()
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_plots_dir = plots_dir / "experiments"
        experiment_plots_dir.mkdir(exist_ok=True)
        experiment_results_dir = results_dir / "experiments"
        experiment_results_dir.mkdir(exist_ok=True)

        plot_path = (
            experiment_plots_dir
            / f"{self.model_name}_{dataset_name}_accs_over_k_{timestamp}.png"
        )
        results_path = (
            experiment_results_dir
            / f"{self.model_name}_{dataset_name}_accs_over_k_{timestamp}.json"
        )

        return plot_path, results_path
    def evaluate_model(
        self, example_selector, k: int, eval_test_set: bool = False
    ):

        # Define the tokens for 'A', 'B', 'C', 'D'
        allowed_tokens = [
            self.tokenizer.convert_tokens_to_ids("A"),
            self.tokenizer.convert_tokens_to_ids("B"),
            self.tokenizer.convert_tokens_to_ids("C"),
            self.tokenizer.convert_tokens_to_ids("D"),
        ]
        logits_processor = LimitTokensLogitsProcessor(allowed_tokens)

        # Initialize lists to store predictions and actual labels
        all_preds = []
        all_labels = []
        accuracy = 0

        _, val_set, test_set = self.dataset.get_data()
        evaluation_set = test_set if eval_test_set else val_set

        print(
            f"Evaluating on {len(evaluation_set)} samples of {'test' if eval_test_set else 'validation'} set"
        )

        for sample in tqdm(evaluation_set):
            examples = example_selector.select_examples(input_variables=sample, key="question", kshot=k)
            few_shot_prompt = self.dataset.create_few_shot_prompt(sample, examples)

            inputs = self.tokenizer(few_shot_prompt, return_tensors="pt").to(self.device)
            # Generate with the custom logits processor
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
                logits_processor=[logits_processor],
            )

            pred_label = self.tokenizer.decode(
                outputs["sequences"][0][-1], skip_special_tokens=True
            ).strip()
            true_label = sample["answerKey"].strip()

            # Store the predictions and true labels
            all_preds.append(pred_label)
            all_labels.append(true_label)

            if pred_label == true_label:
                accuracy += 1

        # Calculate final accuracy
        accuracy = accuracy / len(evaluation_set)
        return accuracy, all_preds, all_labels

    def __repr__(self):
        if not self.dataset and not self.model:
            return (f"Experiments(\n"
                    f"  model_name='{self.model_name}',\n"
                    f"  device='{self.device}',\n"
                    f"  dataset='{self.dataset}',\n"
                    f"  seed={self.args.seed}\n"
                    f"  kshots={self.args.kshots}"
                    f")")
        dataset_sizes = None
        if self.args.portions:
            dataset_sizes = {
                "train": f"{len(self.dataset.train)} ({int(self.args.portions[0]*100)}%)",
                "validation": f"{len(self.dataset.validation)} ({int(self.args.portions[1]*100)}%)",
                "test": f"{len(self.dataset.test)} ({int(self.args.portions[2]*100)}%)",
            }
        elif self.args.sizes:
            dataset_sizes = {
                "train": f"{len(self.dataset.train)} ({self.args.sizes[0]})",
                "validation": f"{len(self.dataset.validation)} ({self.args.sizes[1]})",
                "test": f"{len(self.dataset.test)} ({self.args.sizes[2]})",
            }
        else:
            raise ValueError("Either portions or sizes should be provided")
        return (
            f"Experiments(\n"
            f"  model_name='{self.model_name}',\n"
            f"  device='{self.device}',\n"
            f"  dataset_sizes={dataset_sizes},\n"
            f"  dataset='{self.dataset.get_name()}',\n"
            f"  seed={self.args.seed}\n"
            f"  kshots={self.args.kshots}")
