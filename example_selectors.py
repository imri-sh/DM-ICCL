from abc import ABC, abstractmethod
import random
from typing import Dict, List, Any

from sentence_transformers import SentenceTransformer, util

import utils
from data_mapping import data_mapping, assign_difficulty, device


class BaseExampleSelector(ABC):
    """Interface for selecting examples to include in prompts."""

    @abstractmethod
    def select_examples(self, input_variables: Dict[str, str], key: str, kshot: int) -> List[dict]:
        """Select which examples to use based on the inputs."""

    @abstractmethod
    def add_example(self, example: Dict[str, str]) -> Any:
        """Add new example to store."""


class RandomExampleSelector(BaseExampleSelector):
    def __init__(self, **kwargs):
        self.examples = kwargs['examples']

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables, key, kshot):
        return random.sample(list(self.examples), kshot)


class SimilarityBasedExampleSelector(BaseExampleSelector):

    def __init__(self, **kwargs):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.examples = kwargs['examples']
        self.pool_embeddings = self.compute_embeddings(self.model, self.examples, kwargs['key'])

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables, key, kshot) -> List:
        if kshot == 0:
            return {}
        sample_embedding = self.model.encode([input_variables[key]], convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(sample_embedding, self.pool_embeddings)[0].cpu().numpy()
        top_indices = similarities.argsort()[-kshot:]
        similar_examples = [self.examples[int(i)] for i in top_indices]
        return similar_examples

    @staticmethod
    def compute_embeddings(model, examples, key):
        embeddings = model.encode([example[key] for example in examples], convert_to_tensor=True)
        return embeddings


class DatamapExampleSelector(BaseExampleSelector):
    def __init__(self, **kwargs):
        self.dataset = kwargs['dataset']
        self.model = kwargs['model']
        self.datamap_path = kwargs['datamap_results_path']
        self.datamap_plot_path = kwargs['datamap_plot_path']

        print("---------------------------------------------------------------")
        print(f"Initializing DatamapExampleSelector with the following configuration:\n"
              f"\t Model Name: {kwargs['model_name']}\n"
              f"\t Dataset Name: {self.dataset.get_name()}\n"
              f"\t Number of Evaluations per example: {kwargs['num_evals']}\n"
              f"\t trained with k_shots: {kwargs['datamap_kshots']}"
              f"\t train size: {len(kwargs['dataset'].train)}\n"
              f"\t datamap path: {kwargs['datamap_results_path']}\n"
              f"\t datamap plot path: {kwargs['datamap_plot_path']}")
        print("---------------------------------------------------------------")

        if self.datamap_path is not None and self.datamap_path.exists():
            print(f"The datamap already exists! Loading from {self.datamap_path}")
            self.datamapping_results = utils.load_results(self.datamap_path)
            print(f"{self.datamap_path} loaded successfully.")
        else:
            print("The datamap does not exist. Creating a new one based on the configuration above.")
            self.datamapping_results, _ = data_mapping(
                model=kwargs['model'],
                tokenizer=kwargs['tokenizer'],
                dataset=kwargs['dataset'],
                num_evals=kwargs['num_evals'],
                k_shots=kwargs['datamap_kshots'],
                title=f"{kwargs['model_name']}, {kwargs['dataset'].get_name()} Data Map",
                plot_path=kwargs['datamap_plot_path'],
                show_plot=True
            )
            print(f"Datamap created successfully. saving in {self.datamap_path}")
            if self.datamap_path:
                utils.save_results(
                    self.datamapping_results,
                    save_path=self.datamap_path,
                )
        self.easy, self.ambiguous, self.hard = assign_difficulty(self.datamapping_results)
        print(f"Datamap is ready to use. divided to easy, ambiguous and hard pool examples for ICL.\n"
              f"\tEasy examples pool size: {len(self.easy)}\n"
              f"\tAmbiguous examples pool size: {len(self.ambiguous)}\n"
              f"\tHard pool examples pool size: {len(self.hard)}")

    def add_example(self, example, difficulty='easy'):
        if difficulty == 'easy':
            self.easy.append(example)
        if difficulty == 'ambiguous':
            self.ambiguous.append(example)
        if difficulty == 'hard':
            self.hard.append(example)

    def select_examples(self, input_variables, key, kshot) -> List:
        '''
        :param input_variables:
        :param key:
        :param kshot: [#easy, #ambiguous, #hard] by order
        :return:
        '''
        if kshot == 0:
            return []
        examples = []
        pools = [self.easy, self.ambiguous, self.hard]
        for i in range(len(pools)):
            samples = random.sample(list(pools[i]), kshot)
            samples = [self.dataset.train[sample["sample_index"]] for sample in samples]
            for sample in samples:
                examples.append(sample)
        # print(f"Number of Examples is {len(examples)}")
        return examples


class SimilarityDatamapSelector(BaseExampleSelector):
    def __init__(self, **kwargs):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.examples = kwargs['examples']

        easy_kwargs = kwargs.copy()
        ambiguous_kwargs = kwargs.copy()
        hard_kwargs = kwargs.copy()

        easy_kwargs['examples'] = kwargs["easy_examples"]
        ambiguous_kwargs['examples'] = kwargs["ambiguous_examples"]
        hard_kwargs['examples'] = kwargs["hard_examples"]

        self.similarity_easy = SimilarityBasedExampleSelector(**easy_kwargs)
        self.similarity_ambiguous = SimilarityBasedExampleSelector(**ambiguous_kwargs)
        self.similarity_hard = SimilarityBasedExampleSelector(**hard_kwargs)

    def add_example(self, example):
        raise NotImplementedError

    def select_examples(self, input_variables, key, kshot) -> List:
        assert isinstance(kshot, list) or isinstance(kshot, tuple)
        assert len(kshot) == 3  # easy, ambiguous, hard

        easy_examples = self.similarity_easy.select_examples(input_variables, key, kshot[0])
        ambiguous_examples = self.similarity_ambiguous.select_examples(input_variables, key, kshot[1])
        hard_examples = self.similarity_hard.select_examples(input_variables, key, kshot[2])

        return easy_examples + ambiguous_examples + hard_examples  # TODO - Note, currently orders easy->ambig.->hard


class ExampleSelectorFactory:
    @staticmethod
    def get_example_selector(example_selector_type, **kwargs):
        if example_selector_type == 'random':
            return RandomExampleSelector(**kwargs)
        elif example_selector_type == 'similarity':
            return SimilarityBasedExampleSelector(**kwargs)
        elif example_selector_type == 'datamap':
            return DatamapExampleSelector(**kwargs)
        else:
            raise Exception(f'{example_selector_type} currently not supported.')
