import sklearn
import tensorflow as tf
from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.nn import BayesianModel
from Pyesian.optimizers import HMC
from Pyesian.optimizers.hyperparameters import HyperParameters
from Pyesian.visualisations import Metrics, Plotter
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss
)

np.random.seed(42)

def run_experiment(epsilon, m, L, train_steps=100):
    """
    Builds, trains, and evaluates a classification model on the moons dataset using HMC.
    
    Parameters:
      epsilon (float): Integration step size.
      m (float): Mass parameter.
      L (int): Number of leapfrog steps.
      train_steps (int): Number of training iterations.
      
    Returns:
      acc (float): Test accuracy (in %).
      bayesian_model (BayesianModel): The trained model.
    """
    # Import the moons dataset.
    x, y = sklearn.datasets.make_moons(n_samples=2000, noise=0.2)
    dataset = Dataset(
        tf.data.Dataset.from_tensor_slices((x, y)),
        tf.keras.losses.SparseCategoricalCrossentropy,
        "Classification"
    )
    
    # Build a new Keras model.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(50, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax)
    ])
    
    # Create the Prior distribution.
    prior = GaussianPrior(0.0, -1.0)
    
    # Set HMC hyperparameters.
    hyperparams = HyperParameters(epsilon=epsilon, m=m, L=L)
    
    # Instantiate the HMC optimizer, compile, and train.
    optimizer = HMC()
    optimizer.compile(hyperparams, model.to_json(), dataset, prior=prior)
    optimizer.train(train_steps)
    
    # Retrieve the trained BayesianModel.
    bayesian_model: BayesianModel = optimizer.result()
    
    # Evaluate on the test set.
    test_samples = dataset.test_size
    x_test, y_true = next(iter(dataset.test_data.batch(test_samples)))
    _, preds = bayesian_model.predict(x_test, nb_samples=100)  # Ensemble prediction (averaged)
    preds = preds.numpy() if hasattr(preds, "numpy") else preds
    y_true = y_true.numpy() if hasattr(y_true, "numpy") else y_true
    pred_labels = tf.argmax(preds, axis=1).numpy()
    acc = accuracy_score(y_true, pred_labels) * 100

    # Visualize the decision boundaries and uncertainty area.
    plotter = Plotter(bayesian_model, dataset)
    plotter.plot_decision_boundaries(n_samples=2000, n_boundaries=10)
    
    return acc, bayesian_model

def compute_additional_metrics(bayesian_model):
    """
    Computes evaluation metrics on the test set and for OOD detection.
    
    Returns a dictionary containing:
      - accuracy
      - precision
      - recall
      - f1_score
      - roc_auc
      - brier_score
      - ood_auroc
    """
    # Recreate the moons dataset.
    x, y = sklearn.datasets.make_moons(n_samples=2000, noise=0.2)
    dataset = Dataset(
        tf.data.Dataset.from_tensor_slices((x, y)),
        tf.keras.losses.SparseCategoricalCrossentropy,
        "Classification"
    )
    test_samples = dataset.test_size
    x_test, y_true = next(iter(dataset.test_data.batch(test_samples)))
    _, preds = bayesian_model.predict(x_test, nb_samples=100)
    preds = preds.numpy() if hasattr(preds, "numpy") else preds
    y_true = y_true.numpy() if hasattr(y_true, "numpy") else y_true
    pred_labels = tf.argmax(preds, axis=1).numpy()
    
    # In-distribution (ID) metrics.
    acc = accuracy_score(y_true, pred_labels)
    prec = precision_score(y_true, pred_labels, average='binary')
    rec = recall_score(y_true, pred_labels, average='binary')
    f1 = f1_score(y_true, pred_labels, average='binary')
    roc_auc = roc_auc_score(y_true, preds[:, 1])
    brier = brier_score_loss(y_true, preds[:, 1])
    
    # OOD AUROC: generate OOD data (uniform random samples).
    x_ood = np.random.uniform(low=-2, high=3, size=(1000, 2))
    _, preds_ood = bayesian_model.predict(x_ood, nb_samples=100)
    preds_ood = preds_ood.numpy() if hasattr(preds_ood, "numpy") else preds_ood
    id_scores = np.max(preds, axis=1)       # Maximum softmax probability for ID data.
    ood_scores = np.max(preds_ood, axis=1)    # Maximum softmax probability for OOD data.
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    scores = np.concatenate([id_scores, ood_scores])
    ood_auroc = roc_auc_score(labels, scores)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'brier_score': brier,
        'ood_auroc': ood_auroc
    }

if __name__ == "__main__":
    # Fixed HMC hyperparameters.
    epsilon = 0.005
    m = 0.5
    L = 30
    train_steps = 500
    n_runs = 1

    # List to store metrics from each run.
    metrics_list = []
    
    for run in range(n_runs):
        print(f"\n=== Run {run+1}/{n_runs} ===")
        acc, model = run_experiment(epsilon, m, L, train_steps=train_steps)
        metrics = compute_additional_metrics(model)
        metrics_list.append(metrics)
        print("Run Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    # Aggregate metrics over runs (compute mean and standard deviation).
    keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'brier_score', 'ood_auroc']
    mean_metrics = {}
    std_metrics = {}
    
    print("\n=== Aggregated Metrics over 10 runs ===")
    for key in keys:
        values = [m[key] for m in metrics_list]
        mean_metrics[key] = np.mean(values)
        std_metrics[key] = np.std(values)
        print(f"{key.capitalize()}: {mean_metrics[key]:.4f} ± {std_metrics[key]:.4f}")
