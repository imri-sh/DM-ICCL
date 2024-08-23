import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datasets import load_dataset
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from pathlib import Path
import argparse
import random
from dataset_admin import BaseDataset
from example_selectors import BaseExampleSelector, RandomExampleSelector
from args_utils import get_args, parse_model_arg, prase_dataset_arg, set_seed
import torch
from transformers import LogitsProcessor

plots_dir = Path('./plots')
plots_dir.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_few_shot_prompt(sample, examples):
    prompt = "Choose the correct answers for the following questions, using the letter of the correct answer. Here are some examples: "
    for example in examples:
        prompt += f"Question: {example['question']}\n"
        for i, choice in enumerate(example['choices']['text']):
            prompt += f"{chr(65 + i)}. {choice}\n"
        prompt += f"Answer: {example['answerKey']}\n\n"
    prompt += f"Question: {sample['question']}\n"
    for i, choice in enumerate(sample['choices']['text']):
        prompt += f"{chr(65 + i)}. {choice}\n"
    prompt += "Answer: "
    return prompt


def get_validation_set_and_examples_pool(dataset, pool_size, validation_size):
    # Selecting a pool of examples for few-shot learning
    examples_pool = (
        dataset["train"].select(range(pool_size)) if pool_size > 0 else dataset["train"]
    )
    # Selecting a random validation set
    validation_set = (
        dataset["validation"].select(
            random.sample(range(len(dataset["validation"])), validation_size)
        )
        if validation_size > 0
        else dataset["validation"]
    )
    return examples_pool, validation_set

    # def plot_confusion_matrix(all_labels, all_preds, normalize=False, title='Confusion matrix',
    #                           cmap='Blues', filename="confusion_matrix"):
    #     labels = ["A", "B", "C", "D"]
    #     if normalize:
    #         cm = confusion_matrix(all_labels, all_preds, labels=labels, normalize='true')
    #     else:
    #         cm = confusion_matrix(all_labels, all_preds, labels=labels)
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    #     disp.plot(cmap=cmap, colorbar=False)
    #     plt.title(title)
    #     plt.tight_layout()
    #     filepath = plots_dir / (filename + '.png')
    #     plt.savefig(filepath)
    #     print("Confusion matrix plot saved in", filepath)
    #     plt.show()


class LimitTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_tokens):
        self.allowed_tokens = allowed_tokens

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, -float("inf"))
        mask[:, self.allowed_tokens] = 0
        return scores + mask


def evaluate_model(
    model,
    tokenizer,
    dataset: BaseDataset,
    example_selector: BaseExampleSelector,
    k,
    is_test=False,
):

    # Define the tokens for 'A', 'B', 'C', 'D'
    allowed_tokens = [
        tokenizer.convert_tokens_to_ids("A"),
        tokenizer.convert_tokens_to_ids("B"),
        tokenizer.convert_tokens_to_ids("C"),
        tokenizer.convert_tokens_to_ids("D"),
    ]

    logits_processor = LimitTokensLogitsProcessor(allowed_tokens)

    # Initialize lists to store predictions and actual labels
    all_preds = []
    all_labels = []
    accuracy = 0

    _, val_set, test_set = dataset.get_data()
    eval_set = test_set if is_test else val_set

    print(
        f"Evaluating on {len(eval_set)} samples of {'test' if is_test else 'validation'} set"
    )
    for sample in tqdm(eval_set):
        examples = example_selector.select_examples(
            input_variables=sample, key="question", kshot=k
        )
        few_shot_prompt = dataset.create_few_shot_prompt(sample, examples)

        inputs = tokenizer(few_shot_prompt, return_tensors="pt").to(device)
        # outputs = model.generate(
        #     **inputs,
        #     max_new_tokens=1,
        #     return_dict_in_generate=True,
        #     pad_token_id=tokenizer.eos_token_id,
        # )
        # Generate with the custom logits processor
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=[logits_processor],
        )

        # TODO: make sure pred_label is calculated correctly
        pred_label = tokenizer.decode(
            outputs["sequences"][0][-1], skip_special_tokens=True
        ).strip()
        true_label = sample["answerKey"].strip()

        # Store the predictions and true labels
        all_preds.append(pred_label)
        all_labels.append(true_label)

        if pred_label == true_label:
            accuracy += 1
        # Calculate final accuracy
    accuracy = accuracy / len(eval_set)
    return accuracy, all_preds, all_labels


# def get_filename(args, model_path, accuracy):
#     filename = f'cm_{args.example_selector_type}_{model_path.split("/")[1]}_{args.dataset_name}_kshot_{args.kshot}_acc_{accuracy:.2f}'.replace(
#         '-',
#         '_').replace(
#         '.', '_').lower()
#     return filename


# def get_cm_plot_title(args, model_path, accuracy):
#     title = f"Model: {model_path} \n Dataset: {args.dataset_name}\n With k-shots={args.kshot} \n Accuracy: {accuracy * 100:.2f}%\n Confusion Matrix"
#     return title


# def experiment_models_on_dataset(args, validation_set, example_selector):
#     accs = []
#     for model_path in args.models:
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#         model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
#         accuracy, all_preds, all_labels = evaluate_model(args, model, tokenizer, validation_set, example_selector)
#         plot_confusion_matrix(all_labels,
#                               all_preds,
#                               normalize=False,
#                               title=get_cm_plot_title(args, model_path, accuracy),
#                               filename=get_filename(args, model_path, accuracy)
#                               )
#         accs.append(accuracy)


def experiment_acc_over_k(dataset, ks: list, model_arg, encoder_path=None):
    # accs_rand = []
    # accs_sim = []
    accs = []
    dataset_name = dataset.get_name()

    model, tokenizer, model_name = parse_model_arg(model_arg)

    for k in tqdm(ks, desc="Evaluating model on kshot"):

        # random_example_selector = example_selectors.RandomExampleSelector(examples_pool)
        train_set, _, _ = dataset.get_data()
        example_selector = RandomExampleSelector(train_set)
        accuracy, _, _ = evaluate_model(model, tokenizer, dataset, example_selector, k)
        accs.append(accuracy)

    # random_example_selector = example_selectors.RandomExampleSelector(examples_pool)
    # sim_example_selector = example_selectors.SimilarityBasedExampleSelector(
    #     model_name=encoder_path, examples=examples_pool, key="question"
    # )
    # for k in tqdm(ks, desc="Evaluating model on similarity-based kshot"):
    #     accuracy_sim, _, _ = evaluate_model(
    #         model, tokenizer, validation_set, sim_example_selector, k
    #     )
    #     accs_sim.append(accuracy_sim)
    #     print(f"kshot={k}, accuracy_sim={accuracy_sim * 100:.2f}")

    print(accs)
    return accs
    # for k in tqdm(ks, desc="Evaluating model on random kshot"):
    #     accuracy_rand, _, _ = evaluate_model(
    #         model, tokenizer, validation_set, random_example_selector, k
    #     )
    #     accs_rand.append(accuracy_rand)
    #     print(f"kshot={k}, accuracy_rand={accuracy_rand * 100:.2f}% ")

    # width = 0.2
    # # x =np.arange(max_kshot)
    # # x = np.array(k_range)
    # x = ks
    # plt.bar(x, accs_rand, color='blue', width=width, label="Random Examples", align='center')
    # plt.bar(x + width, accs_sim, color='green', width=width, label="Similarity Based Examples", align='center')
    # plt.legend()
    # plt.xticks([r + width / 2 for r in range(len(x))], x)
    # plt.xlabel("Number of Kshots", fontweight='bold')
    # plt.ylabel("Accuracy")
    # plt.title(f"Model: {model_name} \n Dataset: {dataset_name}")
    # filename = (
    #     f"{model_name}_{dataset_name}_acc_over_k.png".replace("-", "_")
    #     .replace(".", "_")
    #     .lower()
    # )
    # plt.savefig(plots_dir / filename)
    # plt.show()


def trim_data(dataset, portions):
    dataset.train = dataset.train.select(range(int(len(dataset.train) * portions[0])))
    dataset.validation = dataset.validation.select(
        range(int(len(dataset.validation) * portions[1]))
    )
    dataset.test = dataset.test.select(range(int(len(dataset.test) * portions[2])))
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Run the experiment")
    args = get_args(parser).parse_args()

    set_seed(args.seed)

    dataset = prase_dataset_arg(args.dataset)

    portions = args.portions
    # portions = [0.3, 0.3, 0.3]
    if portions is not None:
        dataset = trim_data(dataset, portions)

    ks = [1]
    model_arg = args.model
    encoder_path = args.encoder_path
    experiment_acc_over_k(
        dataset=dataset,
        ks=ks,
        model_arg=model_arg,
        encoder_path=encoder_path,
    )


if __name__ == '__main__':
    main()
