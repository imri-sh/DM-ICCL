from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dataset_admin import ARC_DATASET, BaseDataset
import numpy as np
from tqdm import tqdm
import torch
import random
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt


def main(num_evals: int, k_shots: int, DATASET=ARC_DATASET):
    """
    :param DATASET: The dataset to use - should be an abstract class which implements dataset_admin.BaseDataset.
    :param num_evals: The number of evaluations to do for each sample
    :param k_shots: The number of examples to use as context. Note that the context length of the model limits this.
    """
    # Setting model, tokenizer:
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    dataset = DATASET()  # Change this to switch dataset

    train, train_eval, validation, test = dataset.get_data()
    test = None  # TODO - Currently not touching this.
    # The following commented out section prints out the lengths of the data:
    # print("\ntest - ", type(test), "with length", len(test))
    # print("validation - ", type(validation), "with length", len(validation))
    # print("train - ", type(train), "with length", len(train))
    # print("train_eval - ", type(train_eval), "with length", len(train_eval))

    M = len(train)  # Number of samples to process
    M = 30  # TODO - this is for the sake of testing, overriding the line above (which would use the entire train set)
    train = validation  # TODO - checking performance on train vs on test

    # Store results for each sample
    results = []

    # Run the process num_difficulty_train_samples times
    for sample_idx in tqdm(range(M)):
        sample = train[sample_idx]
        correct_answer = sample['answerKey']
        correct_index = ord(correct_answer) - 65  # Convert 'A' to 0, 'B' to 1, etc.

        correct_probs = []

        for _ in range(num_evals):
            difficulty_samples = random.sample(list(train_eval), k_shots)
            # Create the prompt
            prompt = dataset.create_prompt(sample, difficulty_samples)
            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors='pt')
            # Generate the logits
            outputs = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + 1, output_scores=True,
                                     return_dict_in_generate=True)

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

    # Extract mean and std probabilities
    mean_probs = [result['mean_confidence'] for result in results]
    std_probs = [result['confidence_std'] for result in results]

    print(f"Using M={M}, k={k_shots} and num_evals={num_evals}, "
          f"mean confidence in correct answers is {np.mean(mean_probs)}.")

    plot_datamap(std_probs, mean_probs)


def plot_datamap(std_probs, mean_probs):
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(std_probs, mean_probs, color='blue', alpha=0.6)
    plt.title('Mean vs. Std of Softmax Probabilities of the Correct Answer')
    plt.ylabel('Mean Probability')
    plt.xlabel('Standard Deviation of Probability')
    plt.grid(True)
    plt.xlim(0, 0.2)
    plt.show()


if __name__ == '__main__':
    main(num_evals=2, k_shots=0)
    main(num_evals=4, k_shots=1)
    main(num_evals=4, k_shots=2)
    main(num_evals=4, k_shots=3)
    main(num_evals=4, k_shots=4)
    main(num_evals=4, k_shots=5)
