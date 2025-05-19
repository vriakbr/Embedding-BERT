# main_experiment.py
from extract_embeddings import extract_embeddings
from train_classifier import train_and_evaluate
import numpy as np

strategies = [
    'first', 'last', 'sum_all',
    'second_last', 'sum_last4', 'concat_last4'
]

datasets = {
    'imdb': 'imdb',
    'snli': 'snli'
}

results = {}

for dataset_key, dataset_name in datasets.items():
    print(f"\nRunning for dataset: {dataset_key.upper()}")
    results[dataset_key] = {}
    for strat in strategies:
        print(f" -> Strategy: {strat}")
        extract_embeddings(dataset_name, split='train', strategy=strat, limit=500)

        X = np.load(f'embeddings/{dataset_name}_{strat}_embeddings.npy')
        y = np.load(f'embeddings/{dataset_name}_{strat}_labels.npy')

        print(f"Label distribution: {np.unique(y, return_counts=True)}")

        acc = train_and_evaluate(X, y)
        results[dataset_key][strat] = acc
        print(f"    Accuracy: {acc:.4f}")

print("\nFinal Results:")
for dataset, strat_results in results.items():
    print(f"\n{dataset.upper()}:")
    for strat, score in strat_results.items():
        print(f"{strat:<15}: {score:.4f}")

