import json
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd
data_mapping_jsons_dir = Path("./results/data_mapping_jsons")
results_dir = Path("./results")
# datamaps_dir = Path("./results/datamaps")
plots_dir = Path('./plots')


def plot_confusion_matrix(all_labels, all_preds, normalize=False, title='Confusion matrix',
                          cmap='Blues', filepath:Path=None):
    labels = ["A", "B", "C", "D"]
    if normalize:
        cm = confusion_matrix(all_labels, all_preds, labels=labels, normalize='true')
    else:
        cm = confusion_matrix(all_labels, all_preds, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=cmap, colorbar=False)
    plt.title(title)
    plt.tight_layout()
    filepath = filepath
    if filepath:
        plt.savefig(filepath)
    print("Confusion matrix plot saved in", filepath)
    plt.show()


def plot_accuracies_over_kshots(k_range, accuracies, title,filepath:Path=None):
    x = np.array(k_range)
    plt.bar(x, accuracies, color='blue',width=0.3, label="Examples", align='center')
    plt.legend()
    plt.xticks(k_range)
    plt.xlabel("Number of Kshots", fontweight='bold')
    plt.ylabel("Accuracy")
    plt.title(title)
    if filepath:
        plt.savefig(filepath)
    plt.show()

def plot_datamap(std_probs, mean_probs, filepath: Path=None):
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(std_probs, mean_probs, color='blue', alpha=0.6)
    plt.title('Mean vs. Std of Softmax Probabilities of the Correct Answer')
    plt.ylabel('Mean Probability')
    plt.xlabel('Standard Deviation of Probability')
    plt.grid(True)
    plt.xlim(0, 0.5)
    if filepath:
        plt.savefig(filepath)
    plt.show()


def plot_data_map_by_difficulty(easy, ambiguous, hard, title: str, save_path: Path = None):
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
    plt.title(title)
    plt.xlabel('Confidence Standard Deviation')
    plt.ylabel('Mean Confidence')

    # Add a legend
    plt.legend()

    # Show/save plot
    if save_path:
        plt.savefig(save_path)
    plt.grid(True)
    plt.show()

def trim_data(dataset, portions):
    """
    Trims the dataset by selecting a portion of each subset (train, validation, test).

    :param dataset: The dataset object containing train, validation, and test subsets.
    :param portions: A list or tuple containing the proportions to retain for
                     each subset. Should be in the format [train_portion, validation_portion, test_portion],
                     where each portion is a float between 0 and 1.

    :return: The trimmed dataset with the specified portions of the train, validation, and test subsets.
    """
    if portions == [1.0,1.0,1.0]: return dataset
    # Select portions of each dataset subset
    dataset.train = dataset.train.select(range(int(len(dataset.train) * portions[0])))
    dataset.validation = dataset.validation.select(range(int(len(dataset.validation) * portions[1])))
    dataset.test = dataset.test.select(range(int(len(dataset.test) * portions[2])))
    return dataset

def save_results(results, save_path: Path = None):
    """ Save the results to file."""
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)


def load_results(load_path: Path=None):
    """ Loads the results and returns them."""
    if load_path and load_path.exists():
        with open(load_path, 'r') as f:
            results = json.load(f)
        return results

def plot_experiments(experiments_results_path, plot_path):
    df = pd.read_csv(experiments_results_path)
    # Melt the DataFrame for easier plotting
    df_melted = df.melt(id_vars='kshots', var_name='Model_Dataset', value_name='Accuracy')

    # Plotting with seaborn
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melted, x='kshots', y='Accuracy', hue='Model_Dataset', marker='o')

    # Adding titles and labels
    plt.title('Model Accuracy vs. kshots', fontsize=16)
    plt.xlabel('kshots', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)

    # Enhancing grid and legend
    plt.grid(True)
    plt.legend(title='Model & Dataset', fontsize=12, title_fontsize='13')
    plt.savefig(plot_path)
    # Show plot
    plt.show()