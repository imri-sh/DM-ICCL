from pathlib import Path

import torch
from tqdm import tqdm
from transformers import LogitsProcessor
from utils import plots_dir, results_dir
import utils
from dataset_admin import BaseDataset
from model_loader import ModelLoader
from args_utils import  set_seed, prase_dataset_arg
from example_selectors import RandomExampleSelector, BaseExampleSelector
from datetime import datetime


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
        self.dataset = utils.trim_data(prase_dataset_arg(args.dataset), args.portions)

        self.model, self.tokenizer, self.model_name = ModelLoader.get_model_and_tokenizer(args.model, device=self.device)
        set_seed(args.seed)

    def set_model(self, model_name: str):
        self.model, self.tokenizer, self.model_name = ModelLoader.get_model_and_tokenizer(model_name, device=self.device)

    def set_dataset(self, dataset_name: str, portions=(1.0,1.0,1.0)):
        self.dataset = utils.trim_data(prase_dataset_arg(dataset_name), portions)

    def experiment_acc_over_k(self, ks: list, title: str):
        plot_path, results_path = self.generate_result_paths()
        train_set, _, _ = self.dataset.get_data()
        dataset_name = self.dataset.get_name()
        accs = {}
        example_selector = RandomExampleSelector(train_set)
        for k in tqdm(
            ks,
            desc=f"Evaluating model {self.model_name} on {dataset_name} with kshots {ks}",
        ):
            accuracy, _, _ = self.evaluate_model(example_selector, k)
            accs[k] = accuracy
            print(f"kshot={k}, accuracy_rand={accuracy * 100:.2f}% ")

        utils.plot_accuracies_over_kshots(
            k_range=list(accs.keys()),
            accuracies=list(accs.values()),
            title=title,
            filepath=plot_path,
        )
        utils.save_results(accs, save_path=results_path)

        return accs

    def generate_result_paths(self):
        dataset_name = self.dataset.get_name()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
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
        self, example_selector: BaseExampleSelector, k: int, eval_test_set: bool = False
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
            examples = example_selector.select_examples(
                input_variables=sample, key="question", kshot=k
            )
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
        portions = {"train": f"{len(self.dataset.train)} ({int(self.args.portions[0]*100)}%)",
                    "validation": f"{len(self.dataset.validation)} ({int(self.args.portions[1]*100)}%)",
                    "test": f"{len(self.dataset.test)} ({int(self.args.portions[2]*100)}%)"}

        return (f"Experiments(\n"
                f"  model_name='{self.model_name}',\n"
                f"  device='{self.device}',\n"
                f"  dataset_sizes={portions},\n"
                f"  dataset='{self.args.dataset}',\n"
                f"  seed={self.args.seed}\n"
                f")")


# plots_dir / (filename + '.png')


'''
plots_dir / (
        f"{model_name}_{dataset_name}_accs_over_k.png".replace("-", "_")
        .replace(".", "_")
        .lower()'''
