import tensorflow as tf
from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.nn import BayesianModel
from Pyesian.optimizers import BBB
from Pyesian.optimizers.hyperparameters import HyperParameters
from Pyesian.visualisations import Metrics
import numpy as np
from sklearn.metrics import accuracy_score
from itertools import product

def run_experiment(lr, alpha, batch_size, hidden_dims, train_steps=10000):
    """
    Builds, trains, and evaluates a MNIST classification model using BBB.
    
    Parameters:
      lr (float): Learning rate.
      alpha (float): The alpha hyperparameter for BBB.
      batch_size (int): Batch size.
      hidden_dims (int): Number of neurons in the hidden (dense) layer.
      train_steps (int): Number of training iterations.
    
    Returns:
      acc (float): Test accuracy (in %).
      bayesian_model (BayesianModel): The trained model.
    """
    # Wrap MNIST in the Dataset class.
    dataset = Dataset(
        'mnist',
        tf.keras.losses.SparseCategoricalCrossentropy,
        "Classification"
    )
    
    # Build a new tf.keras model using the hyperparameter for hidden dims.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(hidden_dims, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Create the Prior distribution.
    prior = GaussianPrior(0.0, 1.0)
    
    # Set the BBB hyperparameters.
    hyperparams = HyperParameters(lr=lr, alpha=alpha, batch_size=batch_size)
    
    # Instantiate the BBB optimizer, compile and train the model.
    optimizer = BBB()
    optimizer.compile(hyperparams, model.to_json(), dataset, prior=prior)
    optimizer.train(train_steps)
    
    # Retrieve the trained BayesianModel.
    bayesian_model: BayesianModel = optimizer.result()
    
    # Evaluate on the test set.
    test_samples = dataset.test_size
    x_test, y_true = next(iter(dataset.test_data.batch(test_samples)))
    # Use the model to predict (aggregating predictions if needed)
    _, preds = bayesian_model.predict(x_test, nb_samples=100)  # Assumes single prediction ensemble
    predicted_labels = tf.argmax(preds, axis=1)
    acc = accuracy_score(y_true, predicted_labels) * 100
    
    return acc, bayesian_model

def grid_search(lr_values, alpha_values, batch_size_values, hidden_dims_values, train_steps=10000):
    """
    Performs grid search over hyperparameter ranges for BBB on MNIST classification.
    
    Parameters:
      lr_values (list): List of learning rate values.
      alpha_values (list): List of alpha values.
      batch_size_values (list): List of batch sizes.
      hidden_dims_values (list): List of hidden layer sizes.
      train_steps (int): Number of training steps for each experiment.
      
    Returns:
      best_params (tuple): (lr, alpha, batch_size, hidden_dims) with best accuracy.
      best_acc (float): Best accuracy achieved.
      results (list): List of tuples (lr, alpha, batch_size, hidden_dims, acc) for all experiments.
      best_model (BayesianModel): The model corresponding to the best hyperparameters.
    """
    best_acc = 0.0
    best_params = None
    best_model = None
    results = []
    
    for lr, alpha, bs, hidden in product(lr_values, alpha_values, batch_size_values, hidden_dims_values):
        print(f"Testing: lr={lr}, alpha={alpha}, batch_size={bs}, hidden_dims={hidden}")
        acc, model = run_experiment(lr, alpha, bs, hidden, train_steps=train_steps)
        results.append((lr, alpha, bs, hidden, acc))
        print(f"--> Accuracy: {acc:.2f}%\n")
        if acc > best_acc:
            best_acc = acc
            best_params = (lr, alpha, bs, hidden)
            best_model = model
            
    return best_params, best_acc, results, best_model

if __name__ == "__main__":
    # Define hyperparameter ranges.
    # lr_values = [0.0001]
    # alpha_values = [0.0]
    # batch_size_values = [500]
    # hidden_dims_values = [128]  # Try different hidden layer sizes.
    lr_values = [0.0001, 0.01]
    alpha_values = [0.0]
    batch_size_values = [1000]
    hidden_dims_values = [128, 256]  # Try different hidden layer sizes.
    
    # Run grid search (using 10000 training steps).
    best_params, best_acc, results, best_model = grid_search(
        lr_values, alpha_values, batch_size_values, hidden_dims_values, train_steps=15000
    )
    
    # Write grid search results to a log file.
    with open('logs/BBB_mnist.txt', 'w') as f:
        print("Grid Search Results:", file=f)
        for r in results:
            print(f"lr={r[0]}, alpha={r[1]}, batch_size={r[2]}, hidden_dims={r[3]} => Accuracy: {r[4]:.2f}%", file=f)
        print(f"\nBest hyperparameters: lr={best_params[0]}, alpha={best_params[1]}, batch_size={best_params[2]}, hidden_dims={best_params[3]} with Accuracy: {best_acc:.2f}%", file=f)
    
    # Save the best model.
    best_model.store("bbb-saved")
    
    # Optionally, print metrics summary using the best model.
    metrics = Metrics(best_model, Dataset(
        'mnist',
        tf.keras.losses.SparseCategoricalCrossentropy,
        "Classification"
    ))
    metrics.summary()
