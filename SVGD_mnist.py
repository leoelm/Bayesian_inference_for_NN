import tensorflow as tf
from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.optimizers import SVGD
from Pyesian.optimizers.hyperparameters import HyperParameters
import numpy as np
import sklearn.metrics as skmet

# Wrap it in the Dataset class and indicate your loss
dataset = Dataset(
    'mnist',
    tf.keras.losses.SparseCategoricalCrossentropy,
    "Classification"
)

# Create your tf.keras model
output_dim = 10
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(output_dim, activation='softmax'))

# Create the Prior distribution for the variational posterior network
prior = GaussianPrior(0, 1)
# Indicate your hyperparameters
hyperparams = HyperParameters(lr=0.001, batch_size=1024, M=3)
# Instantiate your optimizer
optimizer = SVGD()
# Provide the optimizer with the training data and training parameters
optimizer.compile(hyperparams, model.to_json(), dataset, prior=prior)
optimizer.train(8000) # if there are any issues, set this to 10000, confirmed works
# You're done ! Here is your trained BayesianModel !
models = optimizer.result()

test_samples = dataset.test_size
agg_preds = np.zeros((test_samples, output_dim))
curr = 0
x,y_true = next(iter(dataset.test_data.batch(test_samples)))
for model in models:
    preds = model.predict(x)
    agg_preds += preds

# print(y_true)
agg_preds /= len(models)
# agg_preds = (agg_preds[:, 0] > agg_preds[:, 1]).astype(int)

accuracy = skmet.accuracy_score(y_true, tf.argmax(agg_preds, axis = 1)) * 100
recall = skmet.recall_score(y_true, tf.argmax(agg_preds, axis = 1), average= "micro") * 100
precision = skmet.precision_score(y_true, tf.argmax(agg_preds, axis = 1), average= "macro") * 100
f1 = skmet.f1_score(y_true, tf.argmax(agg_preds, axis = 1), average = "macro")
# ece = tfp.stats.expected_calibration_error(5, logits = agg_preds, labels_true = y_true)
print(f'Accuracy: {accuracy}%')
print(f'Recall: {recall}%')
print(f'Precision: {precision}%')
print(f'F1 score: {f1}')
# print(f'ECE: {ece}')