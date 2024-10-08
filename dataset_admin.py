from datasets import Dataset, load_dataset
from abc import ABC, abstractmethod

basic_prompt_arc = (
    "Given a question answering task from the 3rd to 9th-grade science exam. "
    "The question contains four options 'A', 'B', 'C' and 'D'."
    "Select the most appropriate choice that answers the question.\n"
)
basic_prompt_agnews = (
    "Classify the news articles into the categories."
    "The categories are labeled 'A', 'B', 'C' and 'D'."
    "Select the most appropriate category for the news article.\n"
)
class BaseDataset(ABC):
    """
    Interface for a dataset for classes which are responsible for:
    1. get_data() - Return the data split into disjoint "difficulty_train, train, validation, test"
    2. create_prompt(sample, difficulty_samples) - Create prompts for models given the question and QA examples.
    """

    @abstractmethod
    def get_data(self) -> tuple[Dataset, Dataset, Dataset]:
        """Returns the dataset split into 3 disjoint parts: train, validation, test"""

    @staticmethod
    @abstractmethod
    def create_few_shot_prompt(sample, context_examples) -> str:
        """
        Given context examples and a sample (a question for the model to answer), creates and returns
        the prompt to be given to the model.
        """

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """Returns the name of the dataset."""


def labels_to_chars(sample):
    if sample['answerKey'].isdigit():
        # Convert numeric labels to letters
        sample['answerKey'] = chr(64 + int(sample['answerKey']))  # '1' -> 'A', '2' -> 'B', etc.
        # Convert each numeric choice in 'choices' to letters
        sample['choices']['text'] = [chr(64 + int(choice)) if choice.isdigit() else choice for choice in
                                     sample['choices']['text']]
    return sample


class ArcDataset(BaseDataset):
    def __init__(self):
        # Load the ARC dataset
        arc_dataset = load_dataset('ai2_arc', 'ARC-Challenge')

        # Preprocess the dataset to handle numeric labels
        arc_dataset = arc_dataset.map(labels_to_chars)

        self.train = arc_dataset["train"]
        self.validation = arc_dataset['validation']
        self.test = arc_dataset['test']

    def get_data(self):
        return self.train, self.validation, self.test

    def create_few_shot_prompt(self, sample, context_examples):
        prompt = basic_prompt_arc
        for example in context_examples:
            prompt += f"Question: {example['question']}\n"
            for i, choice in enumerate(example['choices']['text']):
                prompt += f"{chr(65 + i)}. {choice}\n"
            prompt += f"Answer: {example['answerKey']}\n\n"

        prompt += f"Question: {sample['question']}\n"
        for i, choice in enumerate(sample['choices']['text']):
            prompt += f"{chr(65 + i)}. {choice}\n"
        prompt += "Answer: "
        return prompt

    def get_name(self):
        return "ARC-challenge dataset"


def agnews_preprocessing(sample):
    # Convert numeric labels to letters
    sample["answerKey"] = chr(65 + int(sample["label"]))  # '0' -> 'A', '1' -> 'B', etc.
    sample["question"] = sample["text"]
    sample["choices"] = {
        "text": ["World", "Sports", "Business", "Sci/Tech"],
        "label": ["A", "B", "C", "D"],
    }
    return sample


class AgNewsDataset(BaseDataset):
    def __init__(self):
        # Load the AG News dataset
        ag_news_dataset = load_dataset("ag_news", "default")

        # Preprocess the dataset to handle numeric labels
        ag_news_dataset = ag_news_dataset.map(agnews_preprocessing)

        all_train = ag_news_dataset["train"]
        split = all_train.train_test_split(test_size=0.2)
        self.train = split["train"]
        self.validation = split["test"]
        self.test = ag_news_dataset["test"]

    def get_data(self):
        return self.train, self.validation, self.test

    def create_few_shot_prompt(self, sample, context_examples):
        prompt = basic_prompt_agnews
        for example in context_examples:
            prompt += f"News: {example['question']}\n"
            for i, choice in enumerate(example["choices"]["text"]):
                prompt += f"{chr(65 + i)}. {choice}\n"
            prompt += f"Answer: {example['answerKey']}\n\n"

        prompt += f"News: {sample['question']}\n"
        for i, choice in enumerate(sample["choices"]["text"]):
            prompt += f"{chr(65 + i)}. {choice}\n"

        prompt += "Answer: "
        return prompt

    def get_name(self):
        return "AG News dataset"
