from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import argparse
import random
import example_selectors

plots_dir = Path('./plots')
plots_dir.mkdir(parents=True, exist_ok=True)


def convert_labels(sample):
    if sample['answerKey'].isdigit():
        # Convert numeric labels to letters
        sample['answerKey'] = chr(64 + int(sample['answerKey']))  # '1' -> 'A', '2' -> 'B', etc.
        # Convert each numeric choice in 'choices' to letters
        sample['choices']['text'] = [chr(64 + int(choice)) if choice.isdigit() else choice for choice in
                                     sample['choices']['text']]
    return sample


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


def get_validation_set_and_examples_pool(dataset, pool_size, test_size):
    # Selecting a pool of examples for few-shot learning
    examples_pool = dataset["train"].select(range(pool_size)) if pool_size > 0 else dataset["train"]
    # Selecting a random test set
    if test_size > 0:
        validation_set = dataset["validation"].select(random.sample(range(len(dataset["validation"])), test_size))
    else:
        validation_set = dataset["validation"]
    return examples_pool, validation_set


def plot_confusion_matrix(all_labels, all_preds, normalize=False, title='Confusion matrix',
                          cmap='Blues', filename="confusion_matrix"):
    labels = ["A", "B", "C", "D"]
    if normalize:
        cm = confusion_matrix(all_labels, all_preds, labels=labels, normalize='true')
    else:
        cm = confusion_matrix(all_labels, all_preds, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=cmap, colorbar=False)
    plt.title(title)
    plt.tight_layout()
    filepath = plots_dir / (filename + '.png')
    plt.savefig(filepath)
    print("Confusion matrix plot saved in", filepath)
    plt.show()


def get_args(parser):
    # Adding arguments to the parser
    parser.add_argument("--dataset_path", type=str, default="allenai/ai2_arc", help="Path of the dataset to use")
    parser.add_argument("--dataset_name", type=str, default="ARC-Challenge", help="Name of the dataset to use")
    parser.add_argument("--pool_size", type=int, default=-1,
                        help="Size of examples pool for few shot learning. -1 for maximum size")
    parser.add_argument("--test_size", type=int, default=-1,
                        help="Number of random examples from test set to evaluate. -1 for maximum size")
    parser.add_argument("--kshot", type=int, default=2, help="Number of shots to inject")
    parser.add_argument("--models", type=str, nargs='+', default=["google/flan-t5-base"],
                        help="List of model paths or names")
    parser.add_argument("--seed", type=str, default=42, help="Seed value for random number generator")
    parser.add_argument("--example_selector_type", type=str, default='random',
                        help="The type of example selector to use [sim, random]")
    parser.add_argument("--encoder_path", type=str, default='all-MiniLM-L6-v2', help="The path of the encoder to use")
    return parser


def evaluate_model(args, model, tokenizer, validation_set, example_selector):
    # Initialize lists to store predictions and actual labels
    all_preds = []
    all_labels = []
    accuracy = 0
    for sample in tqdm(validation_set):
        # similar_examples = select_similar_examples(sample, examples_pool, encoder, args.kshot)
        examples = example_selector.select_examples(input_variables=sample,
                                                    key='question',
                                                    kshot=args.kshot)
        few_shot_prompt = create_few_shot_prompt(sample, examples)
        # print(few_shot_prompt)
        inputs = tokenizer(few_shot_prompt, return_tensors='pt')
        outputs = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + 1, output_scores=True,
                                 return_dict_in_generate=True)
        pred_label = tokenizer.decode(outputs['sequences'][0], skip_special_tokens=True).strip()
        true_label = sample["answerKey"].strip()

        # Store the predictions and true labels
        all_preds.append(pred_label)
        all_labels.append(true_label)

        if pred_label == true_label:
            accuracy += 1
        # Calculate final accuracy
    accuracy = accuracy / len(validation_set)
    return accuracy, all_preds, all_labels


def get_filename(args, model_path):
    filename = f'cm_{args.example_selector_type}_{model_path.split("/")[1]}_{args.dataset_name}_kshot_{args.kshot}_acc_{accuracy:.2f}'.replace(
        '-',
        '_').replace(
        '.', '_').lower()
    return filename


def get_cm_plot_title(args, model_path, accuracy):
    title = f"Model: {model_path} \n Dataset: {args.dataset_name}\n With k-shots={args.kshot} \n Accuracy: {accuracy * 100:.2f}%\n Confusion Matrix"
    return title


def main():
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description="Run the experiment")
    parser = get_args(parser)
    # Parsing command-line arguments
    args = parser.parse_args()
    # Setting the seed value for reproducibility
    random.seed(args.seed)
    # Load dataset, examples pool and validation set
    dataset = load_dataset(args.dataset_path, args.dataset_name).map(convert_labels)
    examples_pool, validation_set = get_validation_set_and_examples_pool(dataset, args.pool_size, args.test_size)

    # init example selector instance
    if args.example_selector_type == 'random':
        example_selector = example_selectors.RandomExampleSelector(examples_pool)
    elif args.example_selector_type == 'sim':
        example_selector = example_selectors.SimilarityBasedExampleSelector(model_name=args.encoder_path,
                                                                            examples=examples_pool,
                                                                            key='question')
    accs = []
    for model_path in args.models:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        accuracy, all_preds, all_labels = evaluate_model(args, model, tokenizer, validation_set, example_selector)
        plot_confusion_matrix(all_labels,
                              all_preds,
                              normalize=False,
                              title=get_cm_plot_title(args, model_path, accuracy),
                              filename=get_filename(args, model_path)
                              )
        accs.append(accuracy)


if __name__ == '__main__':
    main()
