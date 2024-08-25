from abc import ABC, abstractmethod
import random
from typing import Dict, List, Any

from sentence_transformers import SentenceTransformer, util

import utils
from data_mapping import data_mapping, assign_difficulty

class BaseExampleSelector(ABC):
    """Interface for selecting examples to include in prompts."""

    @abstractmethod
    def select_examples(self, input_variables: Dict[str, str], key:str, kshot: int) -> List[dict]:
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

    def select_examples(self, input_variables, key, kshot):
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
        self.datamap_path = kwargs['datamap_path']
        self.dataset_name = kwargs['dataset'].get_name()
        self.model_name =  kwargs['model_name']
        self.num_evals = kwargs['num_evals']
        self.datamap_kshots = kwargs['datamap_kshots']
        print(f"Initializing DatamapExampleSelector with the following configuration:\n"
              f"\t Model Name: {self.model_name}\n"
              f"\t Dataset Name: {self.dataset_name}\n"
              f"\t Number of Evaluations per example: {self.num_evals}\n"
              f"\t trained with k_shots: {self.datamap_kshots}")

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
                title=f"{self.model_name}, {self.dataset_name} Data Map",
                plot_path= utils.plots_dir / f"{self.model_name}_{self.dataset_name}_k_{self.datamap_kshots}_num_evals_{self.num_evals}.png",
                show_plot=True
            )
            if self.datamap_path:
                self.datamap_path.mkdir(parents=True, exist_ok=True)
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

    def select_examples(self, input_variables, key, kshot):
        '''
        :param input_variables:
        :param key:
        :param kshots: [#easy, #ambiguous, #hard] by order
        :return:
        '''
        examples = []
        pools = [self.easy, self.ambiguous, self.hard]
        for i in range(len(pools)):
            samples = random.sample(list(pools[i]), kshot)
            examples.append([self.dataset[sample["sample_index"]] for sample in samples])
        return examples

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

