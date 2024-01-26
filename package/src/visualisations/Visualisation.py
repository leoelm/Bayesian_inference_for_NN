import numpy as np
from src.nn.BayesianModel import BayesianModel
from src.datasets.Dataset import Dataset
import sklearn.metrics as met
import tensorflow as tf
import matplotlib.pyplot as plt


class Visualisation():
    def __init__(self, model):
        self.model = model
    
    # https://seaborn.pydata.org
    def visualise(self, dataset: Dataset, nb_samples: int):
        images, labels = tuple(zip(*dataset.valid_data))
        x = tf.transpose(tf.stack(images, axis=1))
        y_true = tf.transpose(tf.stack(labels, axis=1))
        y_samples, y_pred = self.model.predict(x, nb_samples)  # pass in the x value
    
        # Prediction Plot
        plt.figure(figsize=(10, 5))
        plt.scatter(range(len(y_true)), y_true, label='True Values', alpha=0.5)
        plt.scatter(range(len(y_pred)), y_pred, label='Predicted Mean', alpha=0.5)
        plt.legend()
        plt.title('True vs Predicted Values')
        plt.xlabel('Sample Index')
        plt.ylabel('Output')
        plt.show()
        if dataset.get_likelihood_type() == "Regressor":
            self.metrics_regressor(y_pred, y_true)
            epistemic = self.uncertainty_regressor(y_samples)
            
            # uncertainty
            plt.figure(figsize=(10, 5))
            plt.scatter(range(len(epistemic)), epistemic, label='Epistemic Uncertainty', alpha=0.5)
            plt.legend()
            plt.title('Epistemic Uncertainty')
            plt.ylabel('Output')
            plt.show()
            
        elif dataset.get_likelihood_type() == "Classification":
            self.metrics_classification(y_pred, y_true)
            self.uncertainty_classification(y_samples)
        else: 
            print("Invalid loss function")
        
    def metrics_regressor(self, y_pred, y_true):
        mse = met.mean_squared_error(y_true, y_pred)
        rmse = met.mean_squared_error(y_true, y_pred, squared=False)
        mae = met.mean_absolute_error(y_true, y_pred)
        r2 = met.r2_score(y_true, y_pred)
        print("""Performence metrics for Regression:
              Mean Square Error: {}
              Root Mean Square Error: {}
              Mean Absolute Error: {}
              R^2: {}""".format(mse, rmse, mae, r2))
        
    def metrics_classification(self, y_pred, y_true):
        accuracy = met.accuracy_score(y_true, y_pred)
        recall_score = met.recall_score(y_true, y_pred)
        precision = met.precision_score(y_true, y_pred)
        f1 = met.f1_score(y_true, y_pred)
        print("""Performence metrics for Classification:
              Accuracy: {}
              Mean Recall: {}
              Mean Precision: {}
              F1-Score: {}""".format(accuracy, recall_score, precision, f1))
        
    def uncertainty_regressor(self, y_samples) -> tuple:
        variance = np.var(y_samples, axis=0)
        print("""Uncertainty for Regression: 
              Epistemic Uncertainty: {}""".format(variance))
        return variance
        
    def uncertainty_classification(self, y_samples) -> tuple:
        # For classification, we might use the entropy of the predicted probabilities
        # as a measure of aleatoric uncertainty and variance of multiple stochastic
        # forward passes as epistemic uncertainty.

        # Assuming predict returns a distribution over classes for each sample
        mean = np.mean(y_samples, axis=0)
        variance = np.var(y_samples, axis=0)
        print("""Uncertainty for Regression: 
                Epistemic Uncertainty: {}
                Aleatoric Uncertainty: {}""".format(variance, mean))
        
    def learning_diagnostics():
        pass