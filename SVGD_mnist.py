import tensorflow as tf
from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.optimizers import SVGD
from Pyesian.optimizers.hyperparameters import HyperParameters
import numpy as np
import sklearn.metrics as skmet
from itertools import product

def run_experiment(lr, batch_size, M, train_steps=8000):
    """
    Builds, trains, and evaluates a MNIST classification model using SVGD.
    Returns accuracy, recall, precision, and F1 score.
    """
    # Create the Dataset for MNIST classification.
    dataset = Dataset(
        'mnist',
        tf.keras.losses.SparseCategoricalCrossentropy,
        "Classification"
    )
    
    # Build a new tf.keras model.
    output_dim = 10
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    
    # Create the Prior distribution.
    prior = GaussianPrior(0, 1)
    
    # Set the hyperparameters for SVGD.
    hyperparams = HyperParameters(lr=lr, batch_size=batch_size, M=M)
    
    # Instantiate and compile the SVGD optimizer.
    optimizer = SVGD()
    optimizer.compile(hyperparams, model.to_json(), dataset, prior=prior)
    optimizer.train(train_steps)
    
    # Retrieve the ensemble of trained models.
    models = optimizer.result()

    # Evaluate the ensemble on the test set.
    test_samples = dataset.test_size
    agg_preds = np.zeros((test_samples, output_dim))
    x_test, y_true = next(iter(dataset.test_data.batch(test_samples)))
    for m in models:
        preds = m.predict(x_test)
        agg_preds += preds
    agg_preds /= len(models)
    
    # Get predicted labels.
    predicted_labels = tf.argmax(agg_preds, axis=1)
    
    # Compute classification metrics.
    accuracy = skmet.accuracy_score(y_true, predicted_labels) * 100
    recall = skmet.recall_score(y_true, predicted_labels, average="micro") * 100
    precision = skmet.precision_score(y_true, predicted_labels, average="macro") * 100
    f1 = skmet.f1_score(y_true, predicted_labels, average="macro")
    
    return accuracy, recall, precision, f1

def grid_search(lr_values, batch_size_values, M_values, train_steps=8000):
    """
    Performs grid search over the given hyperparameter ranges.
    Returns the best hyperparameter combination (based on accuracy) and all results.
    """
    best_accuracy = 0.0
    best_params = None
    results = []
    
    for lr, bs, M in product(lr_values, batch_size_values, M_values):
        print(f"Testing hyperparameters: lr={lr}, batch_size={bs}, M={M}")
        acc, rec, prec, f1 = run_experiment(lr, bs, M, train_steps=train_steps)
        results.append((lr, bs, M, acc, rec, prec, f1))
        print(f"--> Accuracy: {acc:.2f}%, Recall: {rec:.2f}%, Precision: {prec:.2f}%, F1: {f1:.4f}\n")
        if acc > best_accuracy:
            best_accuracy = acc
            best_params = (lr, bs, M)
            
    return best_params, best_accuracy, results

if __name__ == "__main__":
    # Define hyperparameter ranges to search.
    # lr_values = [0.001]
    # batch_size_values = [1024]
    # M_values = [3]
    lr_values = [0.001, 0.0005, 0.01, 0.1]
    batch_size_values = [1024, 512, 256]
    M_values = [3, 5, 7]
    
    # Run grid search (using 8000 training steps; adjust if needed).
    best_params, best_acc, results = grid_search(lr_values, batch_size_values, M_values, train_steps=10000)
    with open('logs/SVGD_mnist.txt', 'w') as f:
        print("Grid Search Results:", file=f)
        for r in results:
            print(f"lr={r[0]}, batch_size={r[1]}, M={r[2]} => Accuracy: {r[3]:.2f}%, Recall: {r[4]:.2f}%, Precision: {r[5]:.2f}%, F1: {r[6]:.4f}", file=f)
        
        print(f"\nBest hyperparameters: lr={best_params[0]}, batch_size={best_params[1]}, M={best_params[2]} with Accuracy: {best_acc:.2f}%", file=f)
