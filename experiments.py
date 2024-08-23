from pathlib import Path

import torch
from tqdm import tqdm
from transformers import LogitsProcessor

import utils
from dataset_admin import BaseDataset
from model_loader import ModelLoader
from args_utils import  set_seed, prase_dataset_arg
from example_selectors import RandomExampleSelector, BaseExampleSelector



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

    # def _trim_data(self, dataset, portions):
    #     """
    #     Trims the dataset by selecting a portion of each subset (train, validation, test).
    #
    #     :param dataset: The dataset object containing train, validation, and test subsets.
    #     :param portions: A list or tuple containing the proportions to retain for
    #                      each subset. Should be in the format [train_portion, validation_portion, test_portion],
    #                      where each portion is a float between 0 and 1.
    #
    #     :return: The trimmed dataset with the specified portions of the train, validation, and test subsets.
    #     """
    #     if portions == [1.0,1.0,1.0]: return dataset
    #     # Select portions of each dataset subset
    #     dataset.train = dataset.train.select(range(int(len(dataset.train) * portions[0])))
    #     dataset.validation = dataset.validation.select(range(int(len(dataset.validation) * portions[1])))
    #     dataset.test = dataset.test.select(range(int(len(dataset.test) * portions[2])))
    #     return dataset


    def experiment_acc_over_k(self, ks: list, title:str, filepath: Path):
        (trainset, _, _), dataset_name= self.dataset.get_data(), self.dataset.get_name()
        accs = []
        example_selector = RandomExampleSelector(trainset)
        for k in tqdm(ks, desc=f"Evaluating model {self.model_name} on {dataset_name} with kshot"):
            accuracy, _, _ = self.evaluate_model(example_selector, k)
            accs.append(accuracy)
            print(f"kshot={k}, accuracy_rand={accuracy * 100:.2f}% ")

        utils.plot_accuracies_over_kshots(k_range=ks,
                                          accuracies=accs,
                                          title=title,
                                          filepath=filepath)
        return accs


    def evaluate_model(self, example_selector:BaseExampleSelector, k: int, eval_testset:bool=False):

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
        evaluation_set = test_set if eval_testset else val_set

        print(
            f"Evaluating on {len(evaluation_set)} samples of {'test' if eval_testset else 'validation'} set"
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

            # TODO: make sure pred_label is calculated correctly
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