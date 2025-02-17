import sklearn.metrics as skmet
import tensorflow as tf
from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.optimizers import SVGD
from Pyesian.optimizers.hyperparameters import HyperParameters
import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

np.random.seed(42)

def plot_losses(train_losses, val_losses):
    """
    Plots training and validation losses over epochs.
    
    :param train_losses: List of training loss values.
    :param val_losses: List of validation loss values.
    """
    epochs = [i*10 for i in range(1, len(train_losses) + 1)]
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_data(x, y):
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

def evaluate_metrics(y_true, pred_means, pred_stds, alpha=0.95):
    mse = skmet.mean_squared_error(y_true, pred_means)
    rmse = np.sqrt(mse)
    mae = skmet.mean_absolute_error(y_true, pred_means)
    r2 = skmet.r2_score(y_true, pred_means)
    # nlpd = np.mean(0.5 * np.log(2 * np.pi * pred_stds**2) + (y_true - pred_means)**2 / (2 * pred_stds**2))
    ll = np.mean(-0.5 * np.log(2 * np.pi ) - 0.5 * ((y_true - pred_means) ** 2))
    
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "NLPD": 0, "Log-Likelihood": ll,}

def run_experiment(lr, batch_size, M, x, y, train_steps=500):
    dataset = Dataset(
        tf.data.Dataset.from_tensor_slices((x, y)),
        tf.keras.losses.MeanSquaredError,
        "Regression"
    )
    
    output_size = 1
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, activation='linear', input_shape=(1,)),
        tf.keras.layers.Dense(output_size, activation='linear')
    ])
    
    prior = GaussianPrior(0, 1)
    hyperparams = HyperParameters(lr=lr, batch_size=batch_size, M=M)
    optimizer = SVGD()
    optimizer.compile(hyperparams, model.to_json(), dataset, prior=prior)
    
    test_samples = dataset.test_size
    x_test, y_true = next(iter(dataset.test_data.batch(test_samples)))
    optimizer.train(train_steps)
    models, train_losses, valid_losses = optimizer.result()

    plot_losses(train_losses, valid_losses)
    
    predictions = np.array([m.predict(x_test) for m in models])
    pred_means = np.mean(predictions, axis=0)
    pred_stds = np.std(predictions, axis=0)
    
    metrics = evaluate_metrics(y_true, pred_means, pred_stds.flatten())
    return metrics

if __name__ == "__main__":
    x = tf.random.uniform(shape=(600, 1), minval=1, maxval=20, dtype=tf.float32)
    y = 2 * x + 2
    
    runs = 1
    all_metrics = [run_experiment(lr=0.01, batch_size=64, M=10, x=x, y=y, train_steps=1000) for _ in range(runs)]
    
    mean_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}
    std_metrics = {key: np.std([m[key] for m in all_metrics]) for key in all_metrics[0]}
    
    print("Mean Evaluation Metrics:", mean_metrics)
    print("Standard Deviation of Metrics:", std_metrics)
