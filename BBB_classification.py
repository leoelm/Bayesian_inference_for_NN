import sklearn
import tensorflow as tf
from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.nn import BayesianModel
from Pyesian.optimizers import BBB
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
from itertools import product

np.random.seed(42)


def run_experiment(lr, alpha, batch_size, hidden_dims, train_steps=600):
    """
    Builds, trains, and evaluates a classification model on a moons dataset using BBB.
    
    Parameters:
      lr (float): Learning rate.
      alpha (float): The alpha hyperparameter for BBB.
      batch_size (int): Batch size.
      hidden_dims (int): Number of neurons in the hidden layer.
      train_steps (int): Number of training iterations.
    
    Returns:
      acc (float): Test accuracy (in %).
      bayesian_model (BayesianModel): The trained model.
    """
    # Create the moons dataset (without fixed random_state so each run is stochastic)
    x, y = sklearn.datasets.make_moons(n_samples=2000, noise=0.2)
    dataset = Dataset(
        tf.data.Dataset.from_tensor_slices((x, y)),
        tf.keras.losses.SparseCategoricalCrossentropy,
        "Classification"
    )
    
    # Build a new tf.keras model with a variable hidden dimension.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(hidden_dims, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax)
    ])
    
    # Create the Prior distribution
    prior = GaussianPrior(0.0, -1.0)
    
    # Set hyperparameters
    hyperparams = HyperParameters(lr=lr, alpha=alpha, batch_size=batch_size)
    
    # Instantiate BBB optimizer, compile, and train.
    optimizer = BBB()
    optimizer.compile(hyperparams, model.to_json(), dataset, prior=prior)
    optimizer.train(train_steps)
    
    # Retrieve the trained BayesianModel.
    bayesian_model, _, _ = optimizer.result()
    
    # Evaluate on the test set.
    test_samples = dataset.test_size
    x_test, y_true = next(iter(dataset.test_data.batch(test_samples)))
    _, preds = bayesian_model.predict(x_test, nb_samples=100)
    # Convert TensorFlow tensors to NumPy arrays if necessary.
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
    Computes various evaluation metrics on the test set and for OOD detection.
    
    Returns a dictionary containing:
      - accuracy
      - precision
      - recall
      - f1_score
      - roc_auc
      - brier_score
      - ood_auroc
    """
    # Recreate the dataset (same settings as run_experiment)
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
    
    # Classification metrics on in-distribution (ID) data.
    acc = accuracy_score(y_true, pred_labels)
    prec = precision_score(y_true, pred_labels, average='binary')
    rec = recall_score(y_true, pred_labels, average='binary')
    f1 = f1_score(y_true, pred_labels, average='binary')
    roc_auc = roc_auc_score(y_true, preds[:, 1])
    brier = brier_score_loss(y_true, preds[:, 1])
    
    x_ood = np.random.uniform(low=-2, high=3, size=(1000, 2))
    _, preds_ood = bayesian_model.predict(x_ood, nb_samples=100)
    preds_ood = preds_ood.numpy() if hasattr(preds_ood, "numpy") else preds_ood
    
    # Use the maximum softmax probability as the confidence score.
    id_scores = np.max(preds, axis=1)
    ood_scores = np.max(preds_ood, axis=1)
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

def grid_search(lr_values, alpha_values, batch_size_values, hidden_dims_values, train_steps=600, n_runs=10):
    """
    Performs grid search over hyperparameter ranges for BBB, running each configuration n_runs times.
    For each run, additional metrics are computed.
    
    Returns:
      best_params (tuple): (lr, alpha, batch_size, hidden_dims) for best mean accuracy.
      best_acc (float): Best mean accuracy achieved.
      results (list): List of tuples (lr, alpha, batch_size, hidden_dims, mean_metrics, std_metrics)
                      for all configurations.
      best_model (BayesianModel): The trained model corresponding to the best hyperparameters.
    """
    best_acc = 0.0
    best_params = None
    best_model = None
    results = []
    
    # Define the keys we want to aggregate.
    keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'brier_score', 'ood_auroc']
    
    for lr, alpha, bs, hidden in product(lr_values, alpha_values, batch_size_values, hidden_dims_values):
        print(f"Testing: lr={lr}, alpha={alpha}, batch_size={bs}, hidden_dims={hidden}")
        run_metrics_list = []
        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}")
            acc, model = run_experiment(lr, alpha, bs, hidden, train_steps=train_steps)
            metrics = compute_additional_metrics(model)
            run_metrics_list.append(metrics)
        # Compute mean and standard deviation for each metric.
        mean_metrics = {}
        std_metrics = {}
        for key in keys:
            values = [m[key] for m in run_metrics_list]
            mean_metrics[key] = np.mean(values)
            std_metrics[key] = np.std(values)
        results.append((lr, alpha, bs, hidden, mean_metrics, std_metrics))
        
        print("Metrics for this combination:")
        for key in keys:
            print(f"  {key}: {mean_metrics[key]:.4f} ± {std_metrics[key]:.4f}")
        print("")
        
        # Update best model based on mean accuracy.
        if mean_metrics['accuracy'] > best_acc:
            best_acc = mean_metrics['accuracy']
            best_params = (lr, alpha, bs, hidden)
            best_model = model  # from the last run of this combination

            
    return best_params, best_acc, results, best_model

if __name__ == "__main__":
    # Define hyperparameter ranges (single combination in this example)
    lr_values = [0.5]
    alpha_values = [0.0]
    batch_size_values = [128]
    hidden_dims_values = [64]
    
    best_params, best_acc, results, best_model = grid_search(
        lr_values, alpha_values, batch_size_values, hidden_dims_values, train_steps=600, n_runs=1
    )
    
    # Save the best model to a folder.
    best_model.store("bbb-saved")


