from datasets import load_dataset


def convert_labels(sample):
    if sample['answerKey'].isdigit():
        # Convert numeric labels to letters
        sample['answerKey'] = chr(64 + int(sample['answerKey']))  # '1' -> 'A', '2' -> 'B', etc.
        # Convert each numeric choice in 'choices' to letters
        sample['choices']['text'] = [chr(64 + int(choice)) if choice.isdigit() else choice for choice in
                                     sample['choices']['text']]
    return sample


class ARC_DATASET:
    @staticmethod
    def get_data():
        # Load the ARC dataset
        arc_dataset = load_dataset('ai2_arc', 'ARC-Challenge')

        # Preprocess the dataset to handle numeric labels
        arc_dataset = arc_dataset.map(convert_labels)

        # Split the dataset into 33% difficulty-train and 66% train
        train_test_split = arc_dataset['train'].train_test_split(test_size=0.33, seed=42)
        difficulty_train = train_test_split['test']
        train = train_test_split['train']
        validation = arc_dataset['validation']
        test = arc_dataset['test']# TODO - set actual split to three sets

        return difficulty_train, train, validation, test

    @staticmethod
    def create_prompt(sample, difficulty_samples):
        prompt = "Choose the correct answers for the following questions, using the letter of the correct answer.\n\n"
        for ds in difficulty_samples:
            prompt += f"Question: {ds['question']}\n"
            for i, choice in enumerate(ds['choices']['text']):
                prompt += f"{chr(65 + i)}. {choice}\n"
            prompt += f"Answer: {ds['answerKey']}\n\n"

        prompt += f"Question: {sample['question']}\n"
        for i, choice in enumerate(sample['choices']['text']):
            prompt += f"{chr(65 + i)}. {choice}\n"
        prompt += "Answer: "
        return prompt

