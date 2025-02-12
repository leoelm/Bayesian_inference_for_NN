import sklearn.metrics as skmet
import tensorflow as tf
from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.optimizers import SVGD
from Pyesian.optimizers.hyperparameters import HyperParameters
import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

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

# visualize_data(x, y)

# Wrap it in the Dataset class and indicate your loss
dataset = Dataset(
    tf.data.Dataset.from_tensor_slices((x, y)),
    tf.keras.losses.MeanSquaredError,
    "Regression"
)

output_size = 1
# Create your tf.keras model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1, activation='linear', input_shape=(1,)))
model.add(tf.keras.layers.Dense(output_size, activation='linear'))

# Create the Prior distribution for the variational posterior network
prior = GaussianPrior(0, 1)
# Indicate your hyperparameters
hyperparams = HyperParameters(lr=0.1, batch_size=3, M=5)
# Instantiate your optimizer
optimizer = SVGD()
# Provide the optimizer with the training data and training parameters
optimizer.compile(hyperparams, model.to_json(), dataset, prior=prior)
optimizer.train(500)
# You're done ! Here is your trained BayesianModel !
models = optimizer.result()

test_samples = dataset.test_size
agg_preds = np.zeros((test_samples, output_size))
curr = 0
x,y_true = next(iter(dataset.test_data.batch(test_samples)))
for model in models:
    preds = model.predict(x)
    agg_preds += preds

agg_preds /= len(models)

visualize_data(x, agg_preds)

guassian_distribution = tfp.distributions.Normal(tf.cast(y_true, dtype = agg_preds.dtype), tf.ones_like(y_true, dtype = agg_preds.dtype))
log_likelihood = tf.reduce_mean(guassian_distribution.log_prob(agg_preds))

print('MSE:', skmet.mean_squared_error(y_true, agg_preds))
print('RMSE:', skmet.root_mean_squared_error(y_true, agg_preds))
print('MAE:', skmet.mean_absolute_error(y_true, agg_preds))
print("R2 score:", skmet.r2_score(y_true, agg_preds))
print("log likelihood:", log_likelihood.numpy())