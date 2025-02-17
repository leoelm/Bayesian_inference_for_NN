import sklearn
import tensorflow as tf
from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.optimizers import SVGD
from Pyesian.optimizers.hyperparameters import HyperParameters
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    precision_score, 
    roc_auc_score,
    f1_score,
    confusion_matrix,
    brier_score_loss
)
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(y, X, prediction_samples, resolution=200):
    """
    Plots a scatter of the data and overlays decision boundaries from
    multiple model samples.
    """
    # 1) Create a 2D mesh covering the range of X
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # 2) Create the plot and scatter the original data
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], marker='o', c="blue", label="Class 0")
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], marker='x', c="red", label="Class 1")
    
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(f"Multiple Decision Boundaries N={len(prediction_samples)}")

    # 3) For each sample, predict and plot its decision boundary.
    for sample_idx, model_fn in enumerate(prediction_samples, start=1):
        preds = model_fn(grid_points)  # (resolution*resolution, ?)
        Z = tf.reshape(preds[:, 0], xx.shape)
        cs = plt.contour(xx, yy, Z, levels=[0.5], alpha=0.8, colors='r')
    plt.legend(loc="best")
    plt.show()


def compute_ood_auroc(models, id_data, ood_data):
    """
    Compute AUROC for Out-of-Distribution (OOD) detection using softmax scores.
    """
    # Get softmax scores for ID data
    id_preds = np.zeros((len(id_data), 2))
    for model in models:
        id_preds += model.predict(id_data)
    id_preds /= len(models)
    id_scores = np.max(id_preds, axis=1)  # Max softmax probability

    # Get softmax scores for OOD data
    ood_preds = np.zeros((len(ood_data), 2))
    for model in models:
        ood_preds += model.predict(ood_data)
    ood_preds /= len(models)
    ood_scores = np.max(ood_preds, axis=1)

    # Create labels: ID -> 1, OOD -> 0
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    scores = np.concatenate([id_scores, ood_scores])

    # Compute AUROC
    auroc = roc_auc_score(labels, scores)
    print(f"OOD AUROC: {auroc:.4f}")
    return auroc

def compute_classification_metrics(models, x, y):
    """
    Compute several classification metrics on in-distribution data.
    """
    # Ensemble predictions (averaging the softmax outputs)
    ensemble_preds = np.zeros((len(x), 2))
    print(models)
    for model in models:
        ensemble_preds += model.predict(x)
    ensemble_preds /= len(models)
    
    # Predicted labels
    predicted_labels = np.argmax(ensemble_preds, axis=1)
    
    # Compute metrics
    acc = accuracy_score(y, predicted_labels)
    prec = precision_score(y, predicted_labels, average='binary')
    rec = recall_score(y, predicted_labels, average='binary')
    f1 = f1_score(y, predicted_labels, average='binary')
    
    # For ROC AUC and Brier score, use the probability of the positive class (assumed to be index 1)
    auc = roc_auc_score(y, ensemble_preds[:, 1])
    brier = brier_score_loss(y, ensemble_preds[:, 1])
    
    cm = confusion_matrix(y, predicted_labels)
    
    print("\nIn-Distribution Classification Metrics:")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"ROC AUC:     {auc:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "brier_score": brier,
        "confusion_matrix": cm  # This is a 2D array; you can aggregate it separately if desired.
    }

if __name__ == "__main__":
    # Generate in-distribution data once
    x_id, y_id = sklearn.datasets.make_moons(n_samples=2000, noise=0.2, random_state=42)
    
    # Generate OOD data once (set seed for reproducibility)
    np.random.seed(42)
    x_ood = np.random.uniform(low=-2, high=3, size=(1000, 2))
    
    # Lists to store metrics for each run
    metrics_list = []
    ood_aurocs = []
    
    runs = 1
    for run in range(runs):
        print(f"\n{'='*20} Run {run+1}/{runs} {'='*20}")
        
        # Prepare the dataset for training
        dataset = Dataset(
            tf.data.Dataset.from_tensor_slices((x_id, y_id)),
            tf.keras.losses.SparseCategoricalCrossentropy,
            "Classification"
        )
        
        # Define a new model for this run
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        # Set up the optimizer and prior
        prior = GaussianPrior(0, 1)
        lr, batch_size, M, train_steps = 0.001, 64, 10, 1000
        hyperparams = HyperParameters(lr=lr, batch_size=batch_size, M=M)
        optimizer = SVGD()
        optimizer.compile(hyperparams, model.to_json(), dataset, prior=prior)
        
        # Train the ensemble using SVGD
        optimizer.train(train_steps)
        models, _, _= optimizer.result()
        
        # Compute in-distribution classification metrics
        # metrics = compute_classification_metrics(models, x_id, y_id)
        # metrics_list.append(metrics)
        
        # Compute OOD AUROC
        # auroc = compute_ood_auroc(models, x_id, x_ood)
        # ood_aurocs.append(auroc)
        x, y = next(iter(dataset.test_data.batch(dataset.test_size)))
        

        plot_decision_boundary(y.numpy(), x.numpy(), models)
        
        print(f"{'='*60}")
    
    # Aggregate and print mean and standard deviation for each metric over all runs
    metrics_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "brier_score"]
    print("\nAggregated In-Distribution Metrics over {} runs:".format(runs))
    for key in metrics_keys:
        values = [result[key] for result in metrics_list]
        print(f"{key.capitalize()}: mean = {np.mean(values):.4f}, std = {np.std(values):.4f}")
    
    print(f"\nAggregated OOD AUROC over {runs} runs: mean = {np.mean(ood_aurocs):.4f}, std = {np.std(ood_aurocs):.4f}")
