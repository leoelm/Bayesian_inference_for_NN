import sklearn
import tensorflow as tf
from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.optimizers import SVGD
from Pyesian.optimizers.hyperparameters import HyperParameters
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
from itertools import product

def plot_decision_boundary_with_uncertainty(
    X, y, model_fn, resolution=200, lower_thresh=0.1, upper_thresh=0.9
):
    """
    Plots:
      1) A scatter plot of (X, y).
      2) The decision boundary p=0.5 as a contour line.
      3) An 'uncertainty region' where lower_thresh < p < upper_thresh is filled.
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

    # 2) Predict probabilities on the grid.
    # model_fn should return probabilities in [0, 1].
    probs = model_fn(grid_points)
    # Reshape predictions to match the grid (assumes shape (M, 1))
    probs = tf.reshape(probs[:, 0], xx.shape)

    # 3) Plot the data.
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k'
    )
    plt.colorbar(scatter, label="Class Label")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(f"Uncertainty area with threshold={upper_thresh}")

    # 4) Plot the decision boundary at p=0.5.
    plt.contour(xx, yy, probs, levels=[0.5], colors='red', alpha=0.8)

    # 5) Highlight the uncertainty region: lower_thresh < p < upper_thresh.
    uncertainty_mask = (probs > lower_thresh) & (probs < upper_thresh)
    uncertainty_mask = uncertainty_mask.numpy().astype(float)
    plt.contourf(xx, yy, uncertainty_mask, 
                 levels=[0, 0.5, 1],
                 colors=["none", "orange"],
                 alpha=0.3)
    plt.show()


def plot_decision_boundary(y, X, prediction_samples, resolution=200):
    """
    Plots a scatter of the data and overlays decision boundaries from
    multiple model samples.
    """
    # 1) Create a 2D mesh covering the range of X
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
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

    # 3) For each sample, predict and plot its decision boundary.
    for sample_idx, model_fn in enumerate(prediction_samples, start=1):
        preds = model_fn(grid_points)  # (resolution*resolution, ?)
        Z = tf.reshape(preds[:, 0], xx.shape)
        cs = plt.contour(xx, yy, Z, levels=[0.5], alpha=0.8, colors='r')
        if sample_idx == 1:
            cs.collections[0].set_label("Decision Boundary")
    plt.legend(loc="best")
    plt.show()


def run_experiment(lr, batch_size, M, x, y, train_steps=100):
    """
    Builds and trains a classification model with given hyperparameters using SVGD,
    then returns evaluation metrics (accuracy, recall, precision) on the test set.
    """
    # Wrap the data in the Dataset class for classification.
    dataset = Dataset(
        tf.data.Dataset.from_tensor_slices((x, y)),
        tf.keras.losses.SparseCategoricalCrossentropy,
        "Classification"
    )
    
    # Build a new tf.keras model.
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    
    # Create the prior distribution.
    prior = GaussianPrior(0, 1)
    # Set the SVGD hyperparameters.
    hyperparams = HyperParameters(lr=lr, batch_size=batch_size, M=M)
    # Instantiate, compile, and train the SVGD optimizer.
    optimizer = SVGD()
    optimizer.compile(hyperparams, model.to_json(), dataset, prior=prior)
    optimizer.train(train_steps)
    models = optimizer.result()

    # Evaluate the ensemble on the test set.
    test_samples = dataset.test_size
    agg_preds = np.zeros((test_samples, 2))
    x_test, y_true = next(iter(dataset.test_data.batch(test_samples)))
    for m in models:
        preds = m.predict(x_test)
        agg_preds += preds
    agg_preds /= len(models)
    agg_preds = np.argmax(agg_preds, axis=1)
    
    acc = accuracy_score(y_true, agg_preds)
    rec = recall_score(y_true, agg_preds)
    prec = precision_score(y_true, agg_preds)
    return acc, rec, prec


def grid_search(x, y, lr_values, batch_size_values, M_values, train_steps=100):
    """
    Performs grid search over the provided hyperparameter ranges.
    Returns the best hyperparameters (with highest accuracy) along with all results.
    """
    best_acc = 0.0
    best_params = None
    results = []
    
    for lr, bs, M in product(lr_values, batch_size_values, M_values):
        print(f"Testing: lr={lr}, batch_size={bs}, M={M}")
        acc, rec, prec = run_experiment(lr, bs, M, x, y, train_steps=train_steps)
        results.append((lr, bs, M, acc, rec, prec))
        print(f"--> Accuracy: {acc:.4f}, Recall: {rec:.4f}, Precision: {prec:.4f}\n")
        if acc > best_acc:
            best_acc = acc
            best_params = (lr, bs, M)
    
    return best_params, best_acc, results


if __name__ == "__main__":
    # Load a classification dataset from sklearn (e.g., moons with noise).
    x, y = sklearn.datasets.make_moons(n_samples=2000, noise=0.2)
    
    # Define ranges for grid search.
    lr_values = [0.001, 0.01, 0.1, 1.0]
    batch_size_values = [32, 64, 128]
    M_values = [3, 5, 10, 20]
    
    # Run grid search (using a relatively short training for demonstration).
    best_params, best_acc, results = grid_search(x, y, lr_values, batch_size_values, M_values, train_steps=800)
    
    with open('logs/SVGD_classification.txt', 'w') as f:
        print("Grid Search Results:", file=f)
        for r in results:
            print(f"lr={r[0]}, batch_size={r[1]}, M={r[2]} => Accuracy: {r[3]:.4f}, Recall: {r[4]:.4f}, Precision: {r[5]:.4f}", file=f)
        
        print(f"\nBest hyperparameters: lr={best_params[0]}, batch_size={best_params[1]}, M={best_params[2]} with Accuracy={best_acc:.4f}", file=f)
    
    # (Optional) Train a final model using the best hyperparameters.
    if False:
        dataset = Dataset(
            tf.data.Dataset.from_tensor_slices((x, y)),
            tf.keras.losses.SparseCategoricalCrossentropy,
            "Classification"
        )
        
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)))
        model.add(tf.keras.layers.Dense(2, activation='softmax'))
        
        prior = GaussianPrior(0, 1)
        hyperparams = HyperParameters(lr=best_params[0], batch_size=best_params[1], M=best_params[2])
        optimizer = SVGD()
        optimizer.compile(hyperparams, model.to_json(), dataset, prior=prior)
        optimizer.train(100)
        models = optimizer.result()
        
        test_samples = dataset.test_size
        agg_preds = np.zeros((test_samples, 2))
        x_test, y_true = next(iter(dataset.test_data.batch(test_samples)))
        for m in models:
            preds = m.predict(x_test)
            agg_preds += preds
        agg_preds /= len(models)
        agg_preds = np.argmax(agg_preds, axis=1)
        
        # Plot decision boundaries using the best ensemble and uncertainty area using the first model.
        plot_decision_boundary(y, x, models)
        plot_decision_boundary_with_uncertainty(x, y, models[0].predict)
        
        print(f'Final Model Accuracy: {accuracy_score(y_true, agg_preds) * 100:.2f}%')
        print(f'Final Model Recall: {recall_score(y_true, agg_preds) * 100:.2f}%')
        print(f'Final Model Precision: {precision_score(y_true, agg_preds) * 100:.2f}%')
