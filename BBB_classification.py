import sklearn
import tensorflow as tf
from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.nn import BayesianModel
from Pyesian.optimizers import BBB
from Pyesian.optimizers.hyperparameters import HyperParameters
from Pyesian.visualisations import Metrics, Plotter
import numpy as np
from sklearn.metrics import accuracy_score
from itertools import product

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
    # Import dataset from sklearn (moons dataset)
    x, y = sklearn.datasets.make_moons(n_samples=2000, noise=0.2)
    # Wrap it in the Dataset class and indicate your loss
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
    
    # Indicate your hyperparameters
    hyperparams = HyperParameters(lr=lr, alpha=alpha, batch_size=batch_size)
    
    # Instantiate your optimizer and compile the model.
    optimizer = BBB()
    optimizer.compile(hyperparams, model.to_json(), dataset, prior=prior)
    optimizer.train(train_steps)
    
    # Retrieve the trained BayesianModel.
    bayesian_model: BayesianModel = optimizer.result()
    
    # Evaluate on the test set.
    test_samples = dataset.test_size
    x_test, y_true = next(iter(dataset.test_data.batch(test_samples)))
    _, preds = bayesian_model.predict(x_test, nb_samples=100)
    pred_labels = tf.argmax(preds, axis=1)
    acc = accuracy_score(y_true, pred_labels) * 100
    
    return acc, bayesian_model

def grid_search(lr_values, alpha_values, batch_size_values, hidden_dims_values, train_steps=600):
    """
    Performs grid search over hyperparameter ranges for BBB.
    
    Parameters:
      lr_values (list): List of learning rate values.
      alpha_values (list): List of alpha values.
      batch_size_values (list): List of batch sizes.
      hidden_dims_values (list): List of values for the number of hidden neurons.
      train_steps (int): Number of training steps for each experiment.
      
    Returns:
      best_params (tuple): (lr, alpha, batch_size, hidden_dims) for best accuracy.
      best_acc (float): Best accuracy achieved.
      results (list): List of tuples (lr, alpha, batch_size, hidden_dims, acc) for all runs.
      best_model (BayesianModel): The trained model corresponding to best hyperparameters.
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
    # Define hyperparameter ranges
    # lr_values = [0.5]
    # alpha_values = [0.0]
    # batch_size_values = [64]
    # hidden_dims_values = [25]  # Optimize over different hidden dimensions
    lr_values = [0.1, 0.5]
    alpha_values = [0.0, 0.1, 0.5]
    batch_size_values = [64, 128, 512]
    hidden_dims_values = [25, 50, 100]  # Optimize over different hidden dimensions
    
    # Run grid search (using 600 training steps; adjust as needed)
    best_params, best_acc, results, best_model = grid_search(
        lr_values, alpha_values, batch_size_values, hidden_dims_values, train_steps=600
    )
    
    with open('logs/BBB_classification.txt', 'w') as f:
        print("Grid Search Results:", file=f)
        for r in results:
            print(f"lr={r[0]}, alpha={r[1]}, batch_size={r[2]}, hidden_dims={r[3]} => Accuracy: {r[4]:.2f}%", file=f)
    
        print(f"\nBest hyperparameters: lr={best_params[0]}, alpha={best_params[1]}, batch_size={best_params[2]}, hidden_dims={best_params[3]} with Accuracy: {best_acc:.2f}%", file=f)
    
    # Save the best model to a folder
    best_model.store("bbb-saved")
    
    if False:
        # For final evaluation/visualisation, recreate the dataset (ensure consistency)
        x, y = sklearn.datasets.make_moons(n_samples=2000, noise=0.2)
        dataset = Dataset(
            tf.data.Dataset.from_tensor_slices((x, y)),
            tf.keras.losses.SparseCategoricalCrossentropy,
            "Classification"
        )
        
        # Print metrics summary using the best model.
        metrics = Metrics(best_model, dataset)
        metrics.summary()
        
        # Visualize the decision boundaries and uncertainty area.
        plotter = Plotter(best_model, dataset)
        plotter.plot_decision_boundaries(n_samples=100)
        plotter.plot_uncertainty_area(uncertainty_threshold=0.9)
