import sklearn
import tensorflow as tf
from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.optimizers import SVGD
from Pyesian.optimizers.hyperparameters import HyperParameters
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt

def plot_decision_boundary_with_uncertainty(
    X, y, model_fn, resolution=200, lower_thresh=0.1, upper_thresh=0.9
):
    """
    Plots:
      1) A scatter plot of (X, y).
      2) The decision boundary p=0.5 as a contour line.
      3) An 'uncertainty region' where lower_thresh < p < upper_thresh is filled.

    Args:
      X: Input features, shape (N, 2).
      y: Binary labels (0 or 1), shape (N,).
      model_fn: A callable that takes an array of shape (M, 2) and returns 
                predicted probabilities (floats in [0,1]) of shape (M,).
      resolution: Grid resolution for contouring.
      lower_thresh: Lower probability threshold for the uncertainty region.
      upper_thresh: Upper probability threshold for the uncertainty region.
    """
    # 1) Create a grid that spans the data range
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Flatten the grid for prediction
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # 2) Predict probabilities on the grid
    #    model_fn should return probabilities in [0, 1].
    probs = model_fn(grid_points)
    # Reshape to match xx, yy
    probs = tf.reshape(probs[:, 0], xx.shape)

    # 3) Plot the data
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k'
    )
    plt.colorbar(scatter, label="Class Label")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(f"Uncertainty area with threshold={upper_thresh}")

    # 4) Plot the decision boundary at p=0.5
    plt.contour(xx, yy, probs, levels=[0.5], colors='red', alpha=0.8)

    # 5) Highlight the uncertainty region: lower_thresh < p < upper_thresh
    #    First create a mask of points that are "uncertain".
    uncertainty_mask = (probs > lower_thresh) & (probs < upper_thresh)
    uncertainty_mask = uncertainty_mask.numpy()
    # Convert boolean mask to float for contourf
    uncertainty_mask = uncertainty_mask.astype(float)

    # Use contourf to fill the region where the mask = 1
    # We'll have two "levels": 0.0 and 1.0. 
    # The region at 1.0 will be colored, 0.0 will be transparent.
    plt.contourf(xx, yy, uncertainty_mask, 
                 levels=[0, 0.5, 1],  # two regions: [0..0.5) and [0.5..1]
                 colors=["none", "orange"],  # 'none' for the outside, 'orange' for uncertain
                 alpha=0.3)

    plt.show()

def plot_decision_boundary(y, X, prediction_samples, resolution = 200):
# 1) Create a 2D mesh covering the range of X
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    # Flatten the grid so we can feed it to the model
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # 2) Create the plot and scatter the original data
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X[:, 0], X[:, 1],
        c=y, cmap=plt.cm.coolwarm, edgecolors='k'
    )
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(f"Multiple Decision Boundaries N={len(prediction_samples)}")
    plt.colorbar(scatter, label="Class Label")

    # 3) For each "sample" (model variant), predict and plot the decision boundary
    for sample_idx, model_fn in enumerate(prediction_samples, start=1):
        # Model_fn should output probabilities or logits for shape (M, 2) input
        # Here we assume it returns probabilities in [0,1] for the "positive" class
        # If it returns logits, you must apply a sigmoid to get probabilities.
        preds = model_fn(grid_points)  # shape (resolution*resolution,)
        # Reshape to match xx, yy for contour plotting
        Z = tf.reshape(preds[:, 0], xx.shape)
        # Z = preds.reshape(xx.shape)

        # Plot the decision boundary where p=0.5
        # levels=[0.5] -> single contour line
        cs = plt.contour(xx, yy, Z, levels=[0.5], alpha=0.8, colors='r')
        # Optionally label each boundary:
        cs.collections[0].set_label(f"Sample {sample_idx}" if sample_idx == 1 else "")

    # Show the legend for the first boundary line
    plt.legend(loc="best")
    plt.show()

# Import dataset from sklearn
x,y = sklearn.datasets.make_moons(n_samples=2000)
# Wrap it in the Dataset class and indicate your loss
dataset = Dataset(
    tf.data.Dataset.from_tensor_slices((x, y)),
    tf.keras.losses.SparseCategoricalCrossentropy,
    "Classification"
)

# plot_decision_boundary(y, x)

# Create your tf.keras model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)))
model.add(tf.keras.layers.Dense(2, activation='softmax'))

# Create the Prior distribution for the variational posterior network
prior = GaussianPrior(0, 1)
# Indicate your hyperparameters
hyperparams = HyperParameters(lr=0.1, batch_size=100, M=10)
# Instantiate your optimizer
optimizer = SVGD()
# Provide the optimizer with the training data and training parameters
optimizer.compile(hyperparams, model.to_json(), dataset, prior=prior)
optimizer.train(100)
# You're done ! Here is your trained BayesianModel !
models = optimizer.result()

test_samples = dataset.test_size
agg_preds = np.zeros((test_samples, 2))
# curr = 0
x, y = next(iter(dataset.test_data.batch(test_samples)))
for model in models:
    preds = model.predict(x)
    agg_preds += preds

agg_preds /= len(models)
agg_preds = np.argmax(agg_preds, axis=1)

plot_decision_boundary(y, x.numpy(), models)
plot_decision_boundary_with_uncertainty(x.numpy(), y, models[0].predict)
print(f'Accuracy: {accuracy_score(y, agg_preds) * 100}%')
print(f'Recall: {recall_score(y, agg_preds)*100}%')
print(f'Precision: {precision_score(y, agg_preds)*100}%')

# dim1, dim2, grid_x_augmented = self._extract_grid_x(x, base_matrix, granularity, un_zoom_level)
# prediction_samples, _ = self._model.predict(grid_x_augmented, n_boundaries)
# plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], marker='o', c="blue", label="Class 0")
# plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], marker='x', c="red", label="Class 1")
# for pred in prediction_samples:
#     pred = tf.reshape(pred[:, 0], dim1.shape)
#     plt.contour(dim1, dim2, pred, [0.5], colors=["red"])
# plt.legend()
# plt.title("Multiple Decision Boundaries N=" + str(n_boundaries))

# print('MSE:', mean_squared_error(y_true, agg_preds))
# print('RMSE:', root_mean_squared_error(y_true, agg_preds))
# print('MAE:', mean_absolute_error(y_true, agg_preds))
# print("R2 score:", r2_score(y_true, agg_preds))

# # See your metrics and performance
# metrics = Metrics(bayesian_model, dataset)
# metrics.summary()
# # Save your model to a folder
# bayesian_model.store("svgd-saved")