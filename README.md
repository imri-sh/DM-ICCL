Here's a more visually appealing version of the markdown:

# DM-ICCL: Improving In-Context Learning through DataMap-Based Curriculum

This repository contains the implementation of the DM-ICCL framework, which includes three main phases:
- 📊 Datamap Constructions 
- 🏷️ Difficulty Assignment to [easy, ambiguous, hard] subsets 
- 🚀 Application using k-shot curriculum-based context on test set

📑 Our full report can be found [here](https://example.com/path/to/yourfile.pdf).

This repository can be used to build ICL Data-Maps, like this one for ARC-Challenge using Llama3-8B-instruct:

![ARC-Challenge DataMap](pp_datamaps%2Fplots%2Fdm_llama-3-8B-instruct_arc_train_size_1119_k_3_num_evals_5.png)

## 📁 Key Files

- `experiments.py`: Framework for running experiments
- `preprocess_dm.py`: Datamap constructions preprocessing script
- `model_loader.py`: Handles loading and initializing models and tokenizers
- `run_experiments.py`: Main script to run experiments based on provided arguments
- `example_selectors.py`: Implements various example selection strategies
- `dataset_admin.py`: Manages datasets, including loading, pre-processing, and few-shot prompt template creation

## 🛠️ Installation

To use the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/imri-sh/DM-ICCL
cd DM-ICCL
pip install -r requirements.txt
```

## 🚀 Usage

### To replicate our experiments

The following command will execute:

1. Datamap Construction for every (model, dataset) specified
   - Relevant arguments: `train_size`, `datamap_kshots`, `num_evals`, `seed`
   - Saving results in:
     - `./pp_datamaps/plots`
     - `./pp_datamaps/results`
2. Experiments divided into two phases:
   - Baseline experiments (random, similarity)
   - Datamap + datamap_similarity experiments
   - Saving results in:
     - `./results/experiments`

```bash
python run_experiments.py --datasets "arc,agnews" --sizes 1119 299 1172 --models "llama3_8b_instruct,llama_3_8b,phi3_5,gemma2_9b_instruct,gemma2_9b" --datamap_kshots 3 --num_evals 5 --kshots 0 3 --orders "E-A-H,E-H-A,A-E-H,A-H-E,H-E-A,H-A-E" --kshots_datamap_similarity "[[1, 2, 3], [3, 2, 1], [5, 1, 0], [4, 2, 0], [2, 4, 0], [0, 4, 2], [0, 2, 4], [2, 0, 4], [4, 0, 2], [6, 0, 0], [0, 6, 0], [0, 0, 6]]" --seed 42 --eval_test_set
```

See `model_loader.py` and `dataset_admin.py` to view supported models and datasets, and add your own if desired.

For an explanation of the arguments, run:

```bash
python run_experiments.py --help
```