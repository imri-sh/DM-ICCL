from pathlib import Path
import utils
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from model_loader import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_confidence_std(dataset, model, tokenizer, num_evals, k_shots):
    """
    Calculate the mean confidence and standard deviation of softmax probabilities for each sample in the dataset.
    Args:
        dataset (Dataset): The dataset containing the samples.
        model (Model): The model used for generating logits.
        tokenizer (Tokenizer): The tokenizer used for tokenizing the prompt.
        num_evals (int): The number of evaluations to perform for each sample.
        k_shots (int): The number of samples to select for each evaluation.
    Returns:
        list: A list of dictionaries containing the following information for each sample:
            - 'sample_index': The index of the sample.
            - 'mean_confidence': The mean confidence of softmax probabilities.
            - 'confidence_std': The standard deviation of softmax probabilities.
            - 'correct_probs': A list of correct probabilities for each evaluation.
    """
    # Get the training data and its length
    train, _, _ = dataset.get_data()
    train_len = len(train)

    # Store results for each sample
    results = []
    
    # Run the process num_difficulty_train_samples times
    for sample_idx in tqdm(range(train_len)):

        sample = train[sample_idx]
        correct_answer = sample['answerKey']
        correct_index = ord(correct_answer) - 65  # Convert 'A' to 0, 'B' to 1, etc.
        is_llama = type(model).__name__ == "LlamaForCausalLM"
        correct_probs = []

        for _ in range(num_evals):
            # Don't use current example in the k-shots
            train_indices = np.arange(len(train))
            valid_indices = np.delete(train_indices, sample_idx)
            selected_indices = np.random.choice(valid_indices, k_shots, replace=False)
            difficulty_samples = [train[int(i)] for i in selected_indices]

            # Create the prompt
            prompt = dataset.create_few_shot_prompt(sample, difficulty_samples)
            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            # Generate the logits
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
                temperature=(1.5 if is_llama else 1.0),
            )
            # Extract the logits for each answer choice
            answer_logits = {}
            num_choices = len(sample['choices']['text'])
            for i in range(num_choices):
                letter = chr(65 + i)  # Convert index to letter (0 -> 'A', 1 -> 'B', etc.)
                # Llama uses 'Ġ' before each token, so we need to add it here
                token_to_tokenize = f"Ġ{letter}" if is_llama else letter
                token_id = tokenizer.convert_tokens_to_ids(token_to_tokenize)
                # Extract the logits for the token corresponding to the letter
                answer_logits[letter] = outputs.scores[0][0, token_id].item()

            # Convert logits to a tensor
            logits_tensor = torch.tensor([answer_logits[chr(65 + i)] for i in range(num_choices)])

            # Calculate softmax probabilities #TODO - ADD ACCURACY!
            softmax_probs = F.softmax(logits_tensor, dim=0)
            if torch.all(torch.isnan(softmax_probs)).item():
                print(
                    f"Error: Softmax probabilities are all NaN for sample {sample_idx}"
                )
                continue
            # Ensure correct_index is within bounds
            if correct_index >= num_choices or correct_index < 0:
                print(f"Error: Correct index {correct_index} is out of bounds for num choices {num_choices}")
                continue

            # Get the probability of the correct answer and store it
            correct_prob = softmax_probs[correct_index]
            if correct_prob.isnan().item():
                print(
                    f"Debug: Correct prob is NaN for sample {sample_idx}, while other probs are {softmax_probs}"
                )
                correct_prob = torch.tensor(0.0)
            correct_probs.append(correct_prob.item())

        # Calculate mean and standard deviation of the softmax probabilities
        if not correct_probs:
            print(f"Error: No correct probabilities for sample {sample_idx}")
            continue

        mean_confidence = np.mean(correct_probs)
        confidence_std = np.std(correct_probs)

        # Save the results
        results.append({
            'sample_index': sample_idx,
            'mean_confidence': mean_confidence,
            'confidence_std': confidence_std,
            'correct_probs': correct_probs
        })

    return results


def data_mapping(model, tokenizer, dataset, num_evals: int, k_shots: int, title:str, plot_path:Path=None, show_plot=True):
    """
    Evaluates how hard individual samples in the dataset are for the given model. Creates a datamap and returns the
    evaluation results.
    :param title:
    :param show_plot:
    :param plot_path:
    :param model: Model to use for evaluation
    :param tokenizer: The model's tokenizer
    :param dataset: The dataset to use - an instance of an abstract class which implements dataset_admin.BaseDataset
    :param num_evals: The number of evaluations to do for each sample
    :param k_shots: The number of examples to use as context. Note that the context length of the model limits this.
    """
    # Get results for each sample
    # M = None  # TODO - this is for the sake of testing. If M=None, defaults to using entire training set
    print("Making datamap...")
    results = get_confidence_std(dataset, model, tokenizer, num_evals, k_shots)
    # Extract mean and std probabilities
    mean_probs = [result['mean_confidence'] for result in results]
    # std_probs = [result['confidence_std'] for result in results]

    mean_confidence = np.mean(mean_probs)
    easy, ambiguous, hard = assign_difficulty(results)  # Also adds difficulty categories to each example

    if show_plot:
        print(
            f"Using {len(dataset.get_data()[0])} examples with k={k_shots} and num_evals={num_evals}.\n "
            f"Mean confidence in correct answers is {mean_confidence*100:.2f}%"
        )
        utils.plot_data_map_by_difficulty(easy, ambiguous, hard, title=title, save_path=plot_path)
    return results, mean_confidence

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
