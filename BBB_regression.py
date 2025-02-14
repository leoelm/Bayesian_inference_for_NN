import tensorflow as tf
from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.nn import BayesianModel
from Pyesian.optimizers import BBB
from Pyesian.optimizers.hyperparameters import HyperParameters
from Pyesian.visualisations import Metrics, Plotter
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skmet
from itertools import product

def visualize_data(x, y):
    """
    Plots the given x and y data points.
    """
    # Convert tensors to NumPy arrays if necessary.
    x_np = x.numpy().flatten() if isinstance(x, tf.Tensor) else x.flatten()
    y_np = y.numpy().flatten() if isinstance(y, tf.Tensor) else y.flatten()

    plt.figure(figsize=(8, 6))
    plt.scatter(x_np, y_np, color='blue', alpha=0.5, label="Data Points")
    # Plot the best-fit line if data follows a linear trend.
    plt.plot(sorted(x_np), sorted(2*x+2), color='red', linewidth=2, label="Best Fit Line")
    plt.xlabel("x values")
    plt.ylabel("y values")
    plt.title("Data Visualization")
    plt.legend()
    plt.grid(True)
    plt.show()

def run_experiment(lr, alpha, batch_size, hidden_dims, train_steps=2000, n_boundaries=10):
    """
    Builds, trains, and evaluates a regression model on a dummy dataset using BBB.
    
    Parameters:
      lr (float): Learning rate.
      alpha (float): The alpha hyperparameter for BBB.
      batch_size (int): Batch size.
      hidden_dims (int): Number of neurons in the hidden layer.
      train_steps (int): Number of training iterations.
      n_boundaries (int): Number of posterior samples for prediction.
    
    Returns:
      mse (float): Mean squared error on the test set.
      bayesian_model (BayesianModel): The trained model.
    """
    # Create a dummy dataset.
    x = tf.random.uniform(shape=(600, 1), minval=1, maxval=20, dtype=tf.float32)
    y = 2*x + 2
    
    # Wrap data in the Dataset class and indicate your loss.
    dataset = Dataset(
        tf.data.Dataset.from_tensor_slices((x, y)),
        tf.keras.losses.MeanSquaredError,
        "Regression"
    )
    
    # Build a new tf.keras model with variable hidden dimensions.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(hidden_dims, activation='linear', input_shape=(1,)),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    # Create the Prior distribution.
    prior = GaussianPrior(0.0, -1.0)
    
    # Set hyperparameters for BBB.
    hyperparams = HyperParameters(lr=lr, alpha=alpha, batch_size=batch_size)
    
    # Instantiate the BBB optimizer and compile the model.
    optimizer = BBB()
    optimizer.compile(hyperparams, model.to_json(), dataset, prior=prior)
    optimizer.train(train_steps)
    
    # Retrieve the trained BayesianModel.
    bayesian_model: BayesianModel = optimizer.result()
    
    # Evaluate on the test set.
    test_samples = dataset.test_size
    x_test, y_true = next(iter(dataset.test_data.batch(test_samples)))
    # Get ensemble predictions using n_boundaries posterior samples.
    _, y_pred = bayesian_model.predict(x_test, n_boundaries)
    mse = skmet.mean_squared_error(y_true, y_pred)
    
    return mse, bayesian_model

def grid_search(lr_values, alpha_values, batch_size_values, hidden_dims_values, train_steps=2000, n_boundaries=10):
    """
    Performs grid search over hyperparameter ranges for BBB regression.
    
    Parameters:
      lr_values (list): List of learning rate values.
      alpha_values (list): List of alpha values.
      batch_size_values (list): List of batch sizes.
      hidden_dims_values (list): List of values for the number of hidden neurons.
      train_steps (int): Number of training steps for each experiment.
      n_boundaries (int): Number of posterior samples for prediction.
      
    Returns:
      best_params (tuple): (lr, alpha, batch_size, hidden_dims) for best (lowest) MSE.
      best_mse (float): Best (lowest) MSE achieved.
      results (list): List of tuples (lr, alpha, batch_size, hidden_dims, mse) for all runs.
      best_model (BayesianModel): The trained model corresponding to best hyperparameters.
    """
    best_mse = float('inf')
    best_params = None
    best_model = None
    results = []
    
    for lr, alpha, bs, hidden in product(lr_values, alpha_values, batch_size_values, hidden_dims_values):
        print(f"Testing: lr={lr}, alpha={alpha}, batch_size={bs}, hidden_dims={hidden}")
        mse, model = run_experiment(lr, alpha, bs, hidden, train_steps=train_steps, n_boundaries=n_boundaries)
        results.append((lr, alpha, bs, hidden, mse))
        print(f"--> MSE: {mse:.4f}\n")
        if mse < best_mse:
            best_mse = mse
            best_params = (lr, alpha, bs, hidden)
            best_model = model
            
    return best_params, best_mse, results, best_model

if __name__ == "__main__":
    # Define hyperparameter ranges.
    # lr_values = [0.0001]
    # alpha_values = [0.0]
    # batch_size_values = [128]
    # hidden_dims_values = [1]  # Try different hidden layer sizes.
    lr_values = [0.0001, 0.0005, 0.001]
    alpha_values = [0.0, 0.1, 0.5]
    batch_size_values = [128, 512]
    hidden_dims_values = [1, 5, 10]  # Try different hidden layer sizes.
    
    # Run grid search (using 2000 training steps; adjust as needed).
    best_params, best_mse, results, best_model = grid_search(
        lr_values, alpha_values, batch_size_values, hidden_dims_values, train_steps=2000, n_boundaries=10
    )
    
    # Write grid search results to a log file.
    with open('logs/BBB_regression.txt', 'w') as f:
        print("Grid Search Results:", file=f)
        for r in results:
            print(f"lr={r[0]}, alpha={r[1]}, batch_size={r[2]}, hidden_dims={r[3]} => MSE: {r[4]:.4f}", file=f)
        print(f"\nBest hyperparameters: lr={best_params[0]}, alpha={best_params[1]}, batch_size={best_params[2]}, hidden_dims={best_params[3]} with MSE: {best_mse:.4f}", file=f)
    
    # Save the best model.
    best_model.store("model/bbb-regression-saved")
    
    if False:
        # Optionally, visualize the predictions.
        # (The original visualization uses a scatter plot with a best-fit line.)
        test_samples = Dataset(
            tf.data.Dataset.from_tensor_slices((tf.random.uniform(shape=(600,1), minval=1, maxval=20, dtype=tf.float32),
                                                2*tf.random.uniform(shape=(600,1), minval=1, maxval=20, dtype=tf.float32)+2)
            ),
            tf.keras.losses.MeanSquaredError,
            "Regression"
        ).test_size  # Just to get the test size.
        
        x_test, y_true = next(iter(Dataset(
            tf.data.Dataset.from_tensor_slices((tf.random.uniform(shape=(600,1), minval=1, maxval=20, dtype=tf.float32),
                                                2*tf.random.uniform(shape=(600,1), minval=1, maxval=20, dtype=tf.float32)+2)
            ),
            tf.keras.losses.MeanSquaredError,
            "Regression"
        ).test_data.batch(600)))
        
        # Get predictions using the best model.
        _, y_pred = best_model.predict(x_test, n_boundaries=10)
        visualize_data(x_test, y_pred)
