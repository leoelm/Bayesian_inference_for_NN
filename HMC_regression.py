import tensorflow as tf
from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.nn import BayesianModel
from Pyesian.optimizers import HMC
from Pyesian.optimizers.hyperparameters import HyperParameters
from Pyesian.visualisations import Metrics, Plotter
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skmet
from itertools import product

def visualize_data(x, y):
    """
    Plots the given x and y data points.

    Parameters:
      x (tf.Tensor or numpy.ndarray): Input values.
      y (tf.Tensor or numpy.ndarray): Output values.
    """
    # Convert tensors to NumPy arrays if necessary.
    x_np = x.numpy().flatten() if isinstance(x, tf.Tensor) else x.flatten()
    y_np = y.numpy().flatten() if isinstance(y, tf.Tensor) else y.flatten()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x_np, y_np, color='blue', alpha=0.5, label="Data Points")
    # Plot the best-fit line if data follows a linear trend.
    plt.plot(sorted(x_np), sorted(2*x + 2), color='red', linewidth=2, label="Best Fit Line")
    plt.xlabel("x values")
    plt.ylabel("y values")
    plt.title("Data Visualization")
    plt.legend()
    plt.grid(True)
    plt.show()

def run_experiment(epsilon, m, L, train_steps=100, n_boundaries=10):
    """
    Builds, trains, and evaluates a regression model on a dummy dataset using HMC.
    
    Parameters:
      epsilon (float): The integration step size.
      m (float): The mass parameter.
      L (int): Number of leapfrog steps.
      train_steps (int): Number of training iterations.
      n_boundaries (int): Number of posterior samples for prediction.
    
    Returns:
      mse (float): Mean squared error on the test set.
      bayesian_model (BayesianModel): The trained model.
    """
    # Create a dummy regression dataset.
    x = tf.random.uniform(shape=(600, 1), minval=1, maxval=20, dtype=tf.float32)
    y = 2 * x + 2
    
    # Wrap the data in the Dataset class.
    dataset = Dataset(
        tf.data.Dataset.from_tensor_slices((x, y)),
        tf.keras.losses.MeanSquaredError,
        "Regression"
    )
    
    # Build a simple model.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, activation='linear', input_shape=(1,)),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    # Create the Prior distribution.
    prior = GaussianPrior(0.0, -1.0)
    
    # Set the HMC hyperparameters.
    hyperparams = HyperParameters(epsilon=epsilon, m=m, L=L)
    
    # Instantiate and compile the HMC optimizer.
    optimizer = HMC()
    optimizer.compile(hyperparams, model.to_json(), dataset, prior=prior)
    optimizer.train(train_steps)
    
    # Retrieve the trained BayesianModel.
    bayesian_model: BayesianModel = optimizer.result()
    
    # Evaluate the model on the test set.
    test_samples = dataset.test_size
    x_test, y_true = next(iter(dataset.test_data.batch(test_samples)))
    # Get predictions using n_boundaries samples.
    _, y_pred = bayesian_model.predict(x_test, n_boundaries)
    mse = skmet.mean_squared_error(y_true, y_pred)
    
    return mse, bayesian_model

def grid_search(epsilon_values, m_values, L_values, train_steps=100, n_boundaries=10):
    """
    Performs grid search over HMC hyperparameter ranges.

    Parameters:
      epsilon_values (list): List of ε values.
      m_values (list): List of m values.
      L_values (list): List of L (leapfrog steps) values.
      train_steps (int): Number of training iterations for each experiment.
      n_boundaries (int): Number of posterior samples for prediction.
      
    Returns:
      best_params (tuple): (epsilon, m, L) with lowest MSE.
      best_mse (float): Best (lowest) MSE achieved.
      results (list): List of tuples (epsilon, m, L, mse) for all experiments.
      best_model (BayesianModel): The model corresponding to the best hyperparameters.
    """
    best_mse = float('inf')
    best_params = None
    best_model = None
    results = []
    
    for epsilon, m, L in product(epsilon_values, m_values, L_values):
        print(f"Testing: epsilon={epsilon}, m={m}, L={L}")
        mse, model = run_experiment(epsilon, m, L, train_steps=train_steps, n_boundaries=n_boundaries)
        results.append((epsilon, m, L, mse))
        print(f"--> MSE: {mse:.4f}\n")
        if mse < best_mse:
            best_mse = mse
            best_params = (epsilon, m, L)
            best_model = model
            
    return best_params, best_mse, results, best_model

if __name__ == "__main__":
    # Define ranges for HMC hyperparameters.
    # epsilon_values = [0.001]
    # m_values = [1.0]
    # L_values = [10]
    epsilon_values = [0.001, 0.0005, 0.0001]
    m_values = [1.0, 0.5, 2.0]
    L_values = [10, 30, 70]
    
    # Run grid search (using 100 training steps; adjust as needed).
    best_params, best_mse, results, best_model = grid_search(
        epsilon_values, m_values, L_values, train_steps=100, n_boundaries=10
    )
    
    # Write grid search results to a log file.
    with open('logs/HMC_regression.txt', 'w') as f:
        print("Grid Search Results:", file=f)
        for r in results:
            print(f"epsilon={r[0]}, m={r[1]}, L={r[2]} => MSE: {r[3]:.4f}", file=f)
        print(f"\nBest hyperparameters: epsilon={best_params[0]}, m={best_params[1]}, L={best_params[2]} with MSE: {best_mse:.4f}", file=f)
    
    # Save the best model.
    best_model.store("model/hmc-regression-saved")
    
    if False:
        # For visualization, re-create the dummy dataset.
        x_vis = tf.random.uniform(shape=(600, 1), minval=1, maxval=20, dtype=tf.float32)
        y_vis = 2 * x_vis + 2
        # Get predictions using the best model.
        _, y_pred = best_model.predict(x_vis, n_boundaries=10)
        visualize_data(x_vis, y_pred)
