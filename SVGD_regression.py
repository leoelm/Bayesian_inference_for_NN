import sklearn.metrics as skmet
import tensorflow as tf
from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.optimizers import SVGD
from Pyesian.optimizers.hyperparameters import HyperParameters
import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from itertools import product

def visualize_data(x, y):
    """
    Plots the given x and y data points.
    """
    # Convert tensors to NumPy arrays if necessary
    x_np = x.numpy().flatten() if isinstance(x, tf.Tensor) else x.flatten()
    y_np = y.numpy().flatten() if isinstance(y, tf.Tensor) else y.flatten()

    plt.figure(figsize=(8, 6))
    plt.scatter(x_np, y_np, color='blue', alpha=0.5, label="Data Points")
    plt.xlabel("x values")
    plt.ylabel("y values")
    plt.title("Data Visualization")
    plt.legend()
    plt.grid(True)
    plt.show()

def run_experiment(lr, batch_size, M, x, y, train_steps=500):
    """
    Runs training for a given set of hyperparameters and returns the mean squared error (MSE)
    on the test set.
    """
    # Create the Dataset object
    dataset = Dataset(
        tf.data.Dataset.from_tensor_slices((x, y)),
        tf.keras.losses.MeanSquaredError,
        "Regression"
    )
    
    output_size = 1
    # Create a new tf.keras model for this experiment
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(1, activation='linear', input_shape=(1,)))
    model.add(tf.keras.layers.Dense(output_size, activation='linear'))
    
    # Create the Prior distribution for the variational posterior network
    prior = GaussianPrior(0, 1)
    # Set hyperparameters for the SVGD optimizer
    hyperparams = HyperParameters(lr=lr, batch_size=batch_size, M=M)
    # Instantiate and compile the SVGD optimizer
    optimizer = SVGD()
    optimizer.compile(hyperparams, model.to_json(), dataset, prior=prior)
    optimizer.train(train_steps)
    # Retrieve the ensemble of models from SVGD
    models = optimizer.result()

    # Evaluate the ensemble on the test set.
    test_samples = dataset.test_size
    agg_preds = np.zeros((test_samples, output_size))
    x_test, y_true = next(iter(dataset.test_data.batch(test_samples)))
    for m in models:
        preds = m.predict(x_test)
        agg_preds += preds
    agg_preds /= len(models)

    mse = skmet.mean_squared_error(y_true, agg_preds)
    return mse

def grid_search(x, y, lr_values, batch_size_values, M_values, train_steps=500):
    """
    Performs grid search over the given hyperparameter ranges and returns the best combination
    along with all results.
    """
    best_mse = float('inf')
    best_params = None
    results = []
    
    for lr, bs, M in product(lr_values, batch_size_values, M_values):
        print(f"Testing hyperparameters: lr={lr}, batch_size={bs}, M={M}")
        mse = run_experiment(lr, bs, M, x, y, train_steps=train_steps)
        results.append((lr, bs, M, mse))
        print(f"Result: MSE = {mse}\n")
        if mse < best_mse:
            best_mse = mse
            best_params = (lr, bs, M)
    return best_params, best_mse, results

if __name__ == "__main__":
    # Create a dummy dataset
    x = tf.random.uniform(shape=(600, 1), minval=1, maxval=20, dtype=tf.float32)
    y = 2 * x + 2

    # Optionally, visualize the data
    # visualize_data(x, y)

    # Define grid search ranges for the hyperparameters
    lr_values = [0.001, 0.01, 0.1, 1.0]
    batch_size_values = [32, 64, 128]
    M_values = [3, 5, 10, 20]
    
    best_params, best_mse, results = grid_search(x, y, lr_values, batch_size_values, M_values, train_steps=1000)
    
    with open('logs/SVGD_regression.txt', 'w') as f:
        print("\nGrid Search Results:", file=f)
        for r in results:
            print(f"lr={r[0]}, batch_size={r[1]}, M={r[2]} => MSE: {r[3]}", file=f)
        
        print(f"\nBest hyperparameters: lr={best_params[0]}, batch_size={best_params[1]}, M={best_params[2]} with MSE={best_mse}", file=f)
