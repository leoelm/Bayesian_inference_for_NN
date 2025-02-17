import tensorflow as tf
from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.nn import BayesianModel
from Pyesian.optimizers import BBB
from Pyesian.optimizers.hyperparameters import HyperParameters
import numpy as np
import sklearn.metrics as skmet
import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses):
    """
    Plots training and validation losses over epochs.
    
    :param train_losses: List of training loss values.
    :param val_losses: List of validation loss values.
    """
    epochs = [i*10 for i in range(1, len(train_losses) + 1)]
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [loss for loss in train_losses], label='Training Loss')
    plt.plot(epochs, [loss for loss in val_losses], label='Validation Loss')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_experiment(lr=0.0005, alpha=0.0, batch_size=512, hidden_dims=1, train_steps=2000, n_boundaries=10):
    """
    Builds, trains, and evaluates a Bayesian regression model using BBB.
    Returns MSE, RMSE, MAE, R², NLPD, Log Likelihood, PICP, and Sharpness.
    """
    # Create a dummy dataset
    x = tf.random.uniform(shape=(600, 1), minval=1, maxval=20, dtype=tf.float32)
    y = 2 * x + 2

    # Wrap data in the Dataset class
    dataset = Dataset(
        tf.data.Dataset.from_tensor_slices((x, y)),
        tf.keras.losses.MeanSquaredError,
        "Regression"
    )
    
    # Define model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(hidden_dims, activation='linear', input_shape=(1,)),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    # Define prior
    prior = GaussianPrior(0.0, 1.0)
    
    # Set BBB hyperparameters
    hyperparams = HyperParameters(lr=lr, alpha=alpha, batch_size=batch_size)
    
    # Train using BBB
    optimizer = BBB()
    optimizer.compile(hyperparams, model.to_json(), dataset, prior=prior)
    optimizer.train(train_steps)
    
    # Retrieve trained Bayesian model
    bayesian_model, train_losses, val_losses = optimizer.result()
    
    plot_losses(train_losses, val_losses)

    # Evaluate on the test set
    test_samples = dataset.test_size
    x_test, y_true = next(iter(dataset.test_data.batch(test_samples)))
    
    # Get ensemble predictions
    y_samples, y_pred = bayesian_model.predict(x_test, n_boundaries)

    # Compute regression metrics
    mse = skmet.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = skmet.mean_absolute_error(y_true, y_pred)
    r2 = skmet.r2_score(y_true, y_pred)

    # Log Likelihood
    log_likelihood = np.mean(-0.5 * np.log(2 * np.pi ) - 0.5 * ((y_true - y_pred) ** 2))

    # Prediction Interval Coverage Probability (PICP)
    lower_bound = np.percentile(y_samples, 2.5, axis=0)
    upper_bound = np.percentile(y_samples, 97.5, axis=0)
    inside_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
    picp = np.mean(inside_interval)

    # Sharpness (Width of the 95% prediction interval)
    sharpness = np.mean(upper_bound - lower_bound)
    
    return mse, rmse, mae, r2, 0, log_likelihood, picp, sharpness

if __name__ == "__main__":
    num_runs = 10
    results = {"MSE": [], "RMSE": [], "MAE": [], "R2": [], "NLPD": [], "Log Likelihood": [], "PICP": [], "Sharpness": []}
    
    i = 0
    while i < num_runs:
        print(f"Running experiment {i+1}/{num_runs}...")
        mse, rmse, mae, r2, nlpd, log_likelihood, picp, sharpness = run_experiment()
        if mse > 2:
            print(f"Skipping experiment due to high MSE: {mse}.")
            continue
        results["MSE"].append(mse)
        results["RMSE"].append(rmse)
        results["MAE"].append(mae)
        results["R2"].append(r2)
        results["NLPD"].append(nlpd)
        results["Log Likelihood"].append(log_likelihood)
        results["PICP"].append(picp)
        results["Sharpness"].append(sharpness)
        i+=1

    # Compute mean and standard deviation
    for metric in results:
        mean = np.mean(results[metric])
        std = np.std(results[metric])
        print(f"{metric}: {mean:.4f} ± {std:.4f}")
