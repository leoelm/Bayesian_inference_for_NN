import sklearn
import tensorflow as tf
from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.nn import BayesianModel
from Pyesian.optimizers import HMC
from Pyesian.optimizers.hyperparameters import HyperParameters
from Pyesian.visualisations import Metrics, Plotter
import numpy as np
from sklearn.metrics import accuracy_score
from itertools import product

def run_experiment(epsilon, m, L, train_steps=100):
    """
    Builds, trains, and evaluates a MNIST classification model using HMC 
    on the moons dataset.
    
    Parameters:
      epsilon (float): Integration step size.
      m (float): Mass parameter.
      L (int): Number of leapfrog steps.
      train_steps (int): Number of training iterations.
      
    Returns:
      acc (float): Test accuracy (in %).
      bayesian_model (BayesianModel): The trained model.
    """
    # Import the moons dataset.
    x, y = sklearn.datasets.make_moons(n_samples=2000, noise=0.2)
    dataset = Dataset(
        tf.data.Dataset.from_tensor_slices((x, y)),
        tf.keras.losses.SparseCategoricalCrossentropy,
        "Classification"
    )
    
    # Build a new Keras model.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(50, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax)
    ])
    
    # Create the Prior distribution.
    prior = GaussianPrior(0.0, -1.0)
    
    # Set HMC hyperparameters.
    hyperparams = HyperParameters(epsilon=epsilon, m=m, L=L)
    
    # Instantiate the HMC optimizer, compile, and train.
    optimizer = HMC()
    optimizer.compile(hyperparams, model.to_json(), dataset, prior=prior)
    optimizer.train(train_steps)
    
    # Retrieve the trained BayesianModel.
    bayesian_model: BayesianModel = optimizer.result()
    
    # Evaluate on the test set.
    test_samples = dataset.test_size
    x_test, y_true = next(iter(dataset.test_data.batch(test_samples)))
    _, preds = bayesian_model.predict(x_test, nb_samples=100)  # Ensemble prediction (averaged)
    predicted_labels = tf.argmax(preds, axis=1)
    acc = accuracy_score(y_true, predicted_labels) * 100
    
    return acc, bayesian_model

def grid_search(epsilon_values, m_values, L_values, train_steps=100):
    """
    Performs grid search over HMC hyperparameter ranges for the moons classification task.
    
    Parameters:
      epsilon_values (list): List of ε values.
      m_values (list): List of m values.
      L_values (list): List of L values (leapfrog steps).
      train_steps (int): Training iterations per experiment.
      
    Returns:
      best_params (tuple): (epsilon, m, L) with highest accuracy.
      best_acc (float): Best accuracy achieved.
      results (list): List of tuples (epsilon, m, L, acc) for all runs.
      best_model (BayesianModel): The model corresponding to the best hyperparameters.
    """
    best_acc = 0.0
    best_params = None
    best_model = None
    results = []
    
    for epsilon, m, L in product(epsilon_values, m_values, L_values):
        print(f"Testing: epsilon={epsilon}, m={m}, L={L}")
        acc, model = run_experiment(epsilon, m, L, train_steps=train_steps)
        results.append((epsilon, m, L, acc))
        print(f"--> Accuracy: {acc:.2f}%\n")
        if acc > best_acc:
            best_acc = acc
            best_params = (epsilon, m, L)
            best_model = model
            
    return best_params, best_acc, results, best_model

if __name__ == "__main__":
    # Define hyperparameter ranges.
    epsilon_values = [0.01, 0.005, 0.001]
    m_values = [1.0, 0.5, 2.0]
    L_values = [10, 30, 70]
    # epsilon_values = [0.01]
    # m_values = [1.0]
    # L_values = [1]
    
    # Run grid search (adjust train_steps as needed).
    best_params, best_acc, results, best_model = grid_search(epsilon_values, m_values, L_values, train_steps=100)
    
    # Write grid search results to a log file.
    with open('logs/HMC_moons_classification.txt', 'w') as f:
        print("Grid Search Results:", file=f)
        for r in results:
            print(f"epsilon={r[0]}, m={r[1]}, L={r[2]} => Accuracy: {r[3]:.2f}%", file=f)
        print(f"\nBest hyperparameters: epsilon={best_params[0]}, m={best_params[1]}, L={best_params[2]} with Accuracy: {best_acc:.2f}%", file=f)
    
    # Save the best model.
    best_model.store("hmc-moons-saved")
    
    if False:
        # Create a new dataset for final evaluation/visualization.
        x, y = sklearn.datasets.make_moons(n_samples=2000, noise=0.2)
        dataset = Dataset(
            tf.data.Dataset.from_tensor_slices((x, y)),
            tf.keras.losses.SparseCategoricalCrossentropy,
            "Classification"
        )
        
        # Print metrics summary using the best model.
        metrics = Metrics(best_model, dataset)
        metrics.summary()
        
        # Visualize decision boundaries and uncertainty area.
        plotter = Plotter(best_model, dataset)
        plotter.plot_decision_boundaries(n_samples=100)
        plotter.plot_uncertainty_area(uncertainty_threshold=0.9)
