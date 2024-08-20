from example_selectors import RandomExampleSelector
from dataset_admin import ARC_DATASET
import numpy as np
from tqdm import tqdm
import torch
import random
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
from model_loader import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_confidence_std(dataset, model, tokenizer, num_evals, k_shots, M=None):
    """
    # TODO - Document.
    :param dataset:
    :param model:
    :param tokenizer:
    :param M:
    :param num_evals:
    :param k_shots:
    :return:
    """
    train, train_eval, validation, test = dataset.get_data()
    test = None  # TODO - Currently not touching this.

    if M is None:
        M = len(train)
    # Printing out the lengths of the data:
    print("\n")
    print("validation - ", type(validation), "with length", len(validation))
    print("train - ", type(train), "with length", len(train))
    print("train_eval - ", type(train_eval), "with length", len(train_eval))

    # Store results for each sample
    results = []

    # Run the process num_difficulty_train_samples times
    for sample_idx in tqdm(range(M)):
        sample = train[sample_idx]
        correct_answer = sample['answerKey']
        correct_index = ord(correct_answer) - 65  # Convert 'A' to 0, 'B' to 1, etc.

        correct_probs = []

        for _ in range(num_evals):
            difficulty_samples = random.sample(list(train_eval), k_shots)  # TODO - change to example selector?
            # Create the prompt
            prompt = dataset.create_prompt(sample, difficulty_samples)
            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            # Generate the logits
            outputs = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + 1, output_scores=True,
                                     return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)

            # Extract the logits for each answer choice
            answer_logits = {}
            num_choices = len(sample['choices']['text'])
            for i in range(num_choices):
                letter = chr(65 + i)  # Convert index to letter (0 -> 'A', 1 -> 'B', etc.)
                token_id = tokenizer.convert_tokens_to_ids(letter)
                # Extract the logits for the token corresponding to the letter
                answer_logits[letter] = outputs.scores[0][0, token_id].item()

            # Convert logits to a tensor
            logits_tensor = torch.tensor([answer_logits[chr(65 + i)] for i in range(num_choices)])

            # Calculate softmax probabilities
            softmax_probs = F.softmax(logits_tensor, dim=0)

            # Ensure correct_index is within bounds
            if correct_index >= num_choices or correct_index < 0:
                print(f"Error: Correct index {correct_index} is out of bounds for num choices {num_choices}")
                continue

            # Get the probability of the correct answer and store it
            correct_prob = softmax_probs[correct_index]
            correct_probs.append(correct_prob.item())

        # Calculate mean and standard deviation of the softmax probabilities
        mean_confidence = np.mean(correct_probs)
        confidence_std = np.std(correct_probs)

        # Save the results
        results.append({
            'sample_index': sample_idx,
            'mean_confidence': mean_confidence,
            'confidence_std': confidence_std,
            'correct_probs': correct_probs
        })

    # Save results to a file
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to 'results.json'")

    # Load the results from the file
    with open('results.json', 'r') as f:
        results = json.load(f)

    return results


def main(model, tokenizer, num_evals: int, k_shots: int, dataset):
    """
    :param model: Model to use for evaluation
    :param tokenizer: The model's tokenizer
    :param dataset: The dataset to use - an instance of an abstract class which implements dataset_admin.BaseDataset
    :param num_evals: The number of evaluations to do for each sample
    :param k_shots: The number of examples to use as context. Note that the context length of the model limits this.
    """
    # Get results for each sample
    M = 8  # TODO - this is for the sake of testing. If M=None, defaults to using entire training set
    results = get_confidence_std(dataset, model, tokenizer, num_evals, k_shots, M=M)
    # Extract mean and std probabilities
    mean_probs = [result['mean_confidence'] for result in results]
    std_probs = [result['confidence_std'] for result in results]

    print(f"Using M={M}, k={k_shots} and num_evals={num_evals}, "
          f"mean confidence in correct answers is {np.mean(mean_probs)}.")

    easy, ambiguous, hard = assign_difficulty(results)  # Also adds difficulty categories to each example
    plot_data_map_by_difficulty(easy, ambiguous, hard)

    return results


def plot_data_map_by_difficulty(easy, ambiguous, hard):
    # Extract x and y values for each category
    easy_x = [example['confidence_std'] for example in easy]
    easy_y = [example['mean_confidence'] for example in easy]

    ambiguous_x = [example['confidence_std'] for example in ambiguous]
    ambiguous_y = [example['mean_confidence'] for example in ambiguous]

    hard_x = [example['confidence_std'] for example in hard]
    hard_y = [example['mean_confidence'] for example in hard]

    # Create the scatter plot
    plt.figure(figsize=(10, 6))

    plt.scatter(easy_x, easy_y, color='green', label='Easy', alpha=0.6, edgecolors='w', s=100)
    plt.scatter(ambiguous_x, ambiguous_y, color='orange', label='Ambiguous', alpha=0.6, edgecolors='w', s=100)
    plt.scatter(hard_x, hard_y, color='red', label='Hard', alpha=0.6, edgecolors='w', s=100)

    # Plot decision boundary lines
    std_range = np.linspace(0, max(easy_x + ambiguous_x + hard_x), 100)

    # For easy and ambiguous boundary: confidence - 2 * std = 0.5 -> confidence = 0.5 + 2 * std
    easy_boundary_y = 0.5 + 2 * std_range
    plt.plot(std_range, easy_boundary_y, 'b--', label='Easy-Ambiguous Boundary')

    # For ambiguous and hard boundary: confidence + 2 * std = 0.5 -> confidence = 0.5 - 2 * std
    hard_boundary_y = 0.5 - 2 * std_range
    plt.plot(std_range, hard_boundary_y, 'r--', label='Ambiguous-Hard Boundary')

    # Set plot title and labels
    plt.title('Data Map by Difficulty with Decision Boundaries')
    plt.xlabel('Confidence Standard Deviation')
    plt.ylabel('Mean Confidence')

    # Add a legend
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.show()


def assign_difficulty(examples: list[dict]) -> tuple[list, list, list]:
    """
        :param examples: A list of dictionaries, where each dictionary represents an example and includes the
                mean confidence and standard deviation of the example according to some model.
    Splits the given examples to difficulty levels, according to the given model's confidence and it's std.
    Also adds the difficulty level to the "example" dictionary.
    """
    easy = []
    ambiguous = []
    hard = []

    for example in examples:
        confidence = example['mean_confidence']
        std = example['confidence_std']
        if confidence >= 0.5:
            x = confidence - 2 * std
            if x >= 0.5:
                example['difficulty_level'] = 'easy'
                easy.append(example)
            if x < 0.5:
                example['difficulty_level'] = 'ambiguous'
                ambiguous.append(example)
        if confidence < 0.5:
            x = confidence + 2 * std
            if x >= 0.5:
                example['difficulty_level'] = 'ambiguous'
                ambiguous.append(example)
            if x < 0.5:
                example['difficulty_level'] = 'hard'
                hard.append(example)
    return easy, ambiguous, hard


def plot_datamap(std_probs, mean_probs):
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(std_probs, mean_probs, color='blue', alpha=0.6)
    plt.title('Mean vs. Std of Softmax Probabilities of the Correct Answer')
    plt.ylabel('Mean Probability')
    plt.xlabel('Standard Deviation of Probability')
    plt.grid(True)
    plt.xlim(0, 0.5)
    plt.show()


def data_mapping():
    # Get model, tokenizer
    model, tokenizer = get_phi3_5()  # Change the called function to use a different model (see model_loader.py)
    dataset = ARC_DATASET()  # Change the called function to use a different dataset (see dataset_admin.py)
    # Run data mapping:
    # main(model=model, tokenizer=tokenizer, dataset=dataset, num_evals=1, k_shots=0)
    # main(model=model, tokenizer=tokenizer, dataset=dataset, num_evals=5, k_shots=1)
    main(model=model, tokenizer=tokenizer, dataset=dataset, num_evals=5, k_shots=2)
    # main(model=model, tokenizer=tokenizer, dataset=dataset, num_evals=5, k_shots=3)
    # main(model=model, tokenizer=tokenizer, dataset=dataset, num_evals=5, k_shots=4)
    # main(model=model, tokenizer=tokenizer, dataset=dataset, num_evals=5, k_shots=5)


if __name__ == '__main__':
    data_mapping()
