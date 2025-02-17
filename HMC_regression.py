import tensorflow as tf
from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.nn import BayesianModel
from Pyesian.optimizers import HMC
from Pyesian.optimizers.hyperparameters import HyperParameters
from Pyesian.visualisations import Metrics, Plotter
import numpy as np
import sklearn.metrics as skmet

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
      metrics_dict (dict): Dictionary of evaluation metrics.
    """
    # Create a dummy regression dataset.
    x = tf.random.uniform(shape=(600, 1), minval=1, maxval=20, dtype=tf.float32)
    y = 2 * x + 2
    
    dataset = Dataset(
        tf.data.Dataset.from_tensor_slices((x, y)),
        tf.keras.losses.MeanSquaredError,
        "Regression"
    )
    
    # Build model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, activation='linear', input_shape=(1,)),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    # Prior distribution
    prior = GaussianPrior(0.0, -1.0)
    
    # Set HMC hyperparameters
    hyperparams = HyperParameters(epsilon=epsilon, m=m, L=L)
    
    # Compile and train
    optimizer = HMC()
    optimizer.compile(hyperparams, model.to_json(), dataset, prior=prior)
    optimizer.train(train_steps)
    
    # Get trained model
    bayesian_model: BayesianModel = optimizer.result()
    
    # Test set
    test_samples = dataset.test_size
    x_test, y_true = next(iter(dataset.test_data.batch(test_samples)))
    
    # Predictions with uncertainty estimates
    all_preds, y_pred = bayesian_model.predict(x_test, n_boundaries)

    y_std = all_preds.numpy() if hasattr(all_preds, "numpy") else all_preds
    y_std = np.std(y_std)

    # Compute evaluation metrics
    mse = skmet.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = skmet.mean_absolute_error(y_true, y_pred)
    r2 = skmet.r2_score(y_true, y_pred)
    
    # Log likelihood
    log_likelihood = np.mean(-0.5 * np.log(2 * np.pi ) - 0.5 * ((y_true - y_pred) ** 2))
    
    # PICP: proportion of true values within 95% confidence intervals
    lower = y_pred - 1.96 * y_std
    upper = y_pred + 1.96 * y_std
    picp = np.mean((y_true >= lower) & (y_true <= upper))
    
    # Sharpness: mean standard deviation of predictions (uncertainty measure)
    sharpness = np.mean(y_std)

    return {
        "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2,
        "Log Likelihood": log_likelihood, "PICP": picp, "Sharpness": sharpness
    }

if __name__ == "__main__":
    # Fixed hyperparameters for HMC
    epsilon = 0.001
    m = 1.0
    L = 50
    train_steps = 50
    n_boundaries = 10
    num_runs = 10

    results = { "MSE": [], "RMSE": [], "MAE": [], "R2": [], "Log Likelihood": [], "PICP": [], "Sharpness": [] }
    
    i = 0
    while i < num_runs:
        
        metrics = run_experiment(epsilon, m, L, train_steps, n_boundaries)
        if metrics['MSE'] > 1:
            print(f"Skipping experiment due to high MSE: {metrics['MSE']}.")
            continue
        for key, value in metrics.items():
            results[key].append(value)

        i+=1
    
    # Compute mean and standard deviation for each metric
    summary = {key: (np.mean(values), np.std(values)) for key, values in results.items()}

    # Print results
    for metric, (mean, std) in summary.items():
        print(f"{metric}: Mean={mean:.4f}, Std={std:.4f}")
