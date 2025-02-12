import sklearn
import tensorflow as tf
from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.optimizers import SVGD
from Pyesian.optimizers.hyperparameters import HyperParameters
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt

def plot_decision_boundary(y, x):
    # Scatter plot with color-coded labels
    plt.figure(figsize=(8, 6))
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("2D Feature Visualization with Labels")
    plt.colorbar(label="Class Label")
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

test_samples = 200
agg_preds = np.zeros((test_samples, 2))
curr = 0
x,y_true = next(iter(dataset.test_data.batch(test_samples)))
for model in models:
    preds = model.predict(x)
    agg_preds += preds

agg_preds /= len(models)
agg_preds = np.argmax(agg_preds, axis=1)

plot_decision_boundary(agg_preds, x)

print(f'Accuracy: {accuracy_score(y_true, agg_preds) * 100}%')
print(f'Recall: {recall_score(y_true, agg_preds)*100}%')
print(f'Precision: {precision_score(y_true, agg_preds)*100}%')

# print('MSE:', mean_squared_error(y_true, agg_preds))
# print('RMSE:', root_mean_squared_error(y_true, agg_preds))
# print('MAE:', mean_absolute_error(y_true, agg_preds))
# print("R2 score:", r2_score(y_true, agg_preds))

# # See your metrics and performance
# metrics = Metrics(bayesian_model, dataset)
# metrics.summary()
# # Save your model to a folder
# bayesian_model.store("svgd-saved")