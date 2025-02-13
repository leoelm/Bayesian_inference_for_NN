import tensorflow as tf
from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.nn import BayesianModel
from Pyesian.optimizers import BBB
from Pyesian.optimizers.hyperparameters import HyperParameters
from Pyesian.visualisations import Metrics, Plotter
import matplotlib.pyplot as plt
import numpy as np

def visualize_data(x, y):
    """
    Plots the given x and y data points.

    Parameters:
    x (tf.Tensor or numpy.ndarray): Input values
    y (tf.Tensor or numpy.ndarray): Output values
    """
    # Convert tensors to NumPy arrays if necessary
    x_np = x.numpy().flatten() if isinstance(x, tf.Tensor) else x.flatten()
    y_np = y.numpy().flatten() if isinstance(y, tf.Tensor) else y.flatten()

    # Plot the dataset
    plt.figure(figsize=(8, 6))
    plt.scatter(x_np, y_np, color='blue', alpha=0.5, label="Data Points")
    
    # Plot the best-fit line if data follows a linear trend
    plt.plot(sorted(x_np), sorted(2*x+2), color='red', linewidth=2, label="Best Fit Line")
    
    plt.xlabel("x values")
    plt.ylabel("y values")
    plt.title("Data Visualization")
    plt.legend()
    plt.grid(True)
    plt.show()

# Create a dummy dataset
x = tf.random.uniform(shape=(600,1), minval=1, maxval=20, dtype=tf.float32)
y = 2*x+2
# Wrap it in the Dataset class and indicate your loss
dataset = Dataset(
    tf.data.Dataset.from_tensor_slices((x, y)),
    tf.keras.losses.MeanSquaredError,
    "Regression"
)

# Create your tf.keras model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(5, activation='linear', input_shape=(1,)))
model.add(tf.keras.layers.Dense(1, activation='linear'))

# Create the Prior distribution
prior = GaussianPrior(0.0, 1.0)
# Indicate your hyperparameters
hyperparams = HyperParameters(lr=0.0001, alpha=0.0, batch_size=1000)
# Instantiate your optimizer
optimizer = BBB()
# Compile the optimizer with your data and the training parameters
optimizer.compile(hyperparams, model.to_json(), dataset, prior=prior)
optimizer.train(2000)
# You are done! Here is your BayesianModel
bayesian_model: BayesianModel = optimizer.result()

# See your metrics and performance
metrics = Metrics(bayesian_model, dataset)
metrics.summary()
# Save your model to a folder
bayesian_model.store("model/bbb-regression-saved")

# plotter = Plotter(bayesian_model, dataset)
# plotter.compare_prediction_to_target()
n_boundaries = 10

test_samples = dataset.test_size
agg_preds = np.zeros((test_samples, 1))

x,y_true = next(iter(dataset.test_data.batch(test_samples)))
y_samples, y_pred = bayesian_model.predict(x, n_boundaries)

visualize_data(x, y_pred)