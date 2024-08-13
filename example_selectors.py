from abc import ABC, abstractmethod
import random
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer, util


class BaseExampleSelector(ABC):
    """Interface for selecting examples to include in prompts."""

    @abstractmethod
    def select_examples(self, input_variables: Dict[str, str], key:str, kshot: int) -> List[dict]:
        """Select which examples to use based on the inputs."""

    @abstractmethod
    def add_example(self, example: Dict[str, str]) -> Any:
        """Add new example to store."""


class RandomExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        self.examples = examples

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables, key, kshot):
        return random.sample(list(self.examples), kshot)


class SimilarityBasedExampleSelector(BaseExampleSelector):
    def __init__(self, model_name, examples,key):
        self.model = SentenceTransformer(model_name)
        self.examples = examples
        self.pool_embeddings = self.compute_embeddings(self.model, examples, key)

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables, key, kshot):
        sample_embedding = self.model.encode([input_variables[key]], convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(sample_embedding, self.pool_embeddings)[0].cpu().numpy()
        top_indices = similarities.argsort()[-kshot:]
        similar_examples = [self.examples[int(i)] for i in top_indices]
        return similar_examples

    def compute_embeddings(self, model, examples, key):
        embeddings = model.encode([example[key] for example in examples], convert_to_tensor=True)
        return embeddings
