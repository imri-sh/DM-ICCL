from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dataset_admin import ARC_DATASET
import numpy as np
from tqdm import tqdm
import torch
import random
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt


def main():
    # Setting model, tokenizer:
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    dataset = ARC_DATASET()  # Change this to switch dataset
    difficulty_train, train, test = dataset.get_data()  # TODO - currently test is None. Need to decide on split.


    M = len(train)  # Number of samples to process
    M = 1  # TODO - this is for the sake of testing, overriding the line above (which would use the entire train set)
    num_difficulty_train_samples = 5  # Number of times to run the process for each sample
    k = 3  # Number of difficulty samples for in-context learning, i.e. k-shot

    # Store results for each sample
    results = []

    # Run the process num_difficulty_train_samples times
    for sample_idx in tqdm(range(M)):
        sample = train[sample_idx]
        correct_answer = sample['answerKey']
        correct_index = ord(correct_answer) - 65  # Convert 'A' to 0, 'B' to 1, etc.

        correct_probs = []

        for _ in range(num_difficulty_train_samples):
            difficulty_samples = random.sample(list(difficulty_train), k)
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
        mean_prob = np.mean(correct_probs)
        std_prob = np.std(correct_probs)

        # Save the results
        results.append({
            'sample_index': sample_idx,
            'mean_prob': mean_prob,
            'std_prob': std_prob,
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
    mean_probs = [result['mean_prob'] for result in results]
    std_probs = [result['std_prob'] for result in results]

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
    main()
