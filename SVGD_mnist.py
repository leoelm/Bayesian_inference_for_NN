import tensorflow as tf
from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.optimizers import SVGD
from Pyesian.optimizers.hyperparameters import HyperParameters
import numpy as np
import sklearn.metrics as skmet

np.random.seed(42)

def run_experiment(lr=0.01, batch_size=1024, M=3, train_steps=10000):
    """
    Builds, trains, and evaluates an MNIST classification model using SVGD.
    Returns accuracy, recall, precision, F1 score, MNIST ROC AUC, and OOD AUROC (using FashionMNIST).
    """
    # ----- Train on MNIST -----
    # Load MNIST dataset (via Pyesian)
    dataset = Dataset(
        'mnist',
        tf.keras.losses.SparseCategoricalCrossentropy,
        "Classification"
    )
    
    # Define a simple classification model for MNIST
    output_dim = 10
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    
    # Define a prior for SVGD
    prior = GaussianPrior(0, 1)
    
    # Set SVGD hyperparameters
    hyperparams = HyperParameters(lr=lr, batch_size=batch_size, M=M)
    
    # Train using SVGD
    optimizer = SVGD()
    optimizer.compile(hyperparams, model.to_json(), dataset, prior=prior)
    optimizer.train(train_steps)
    
    # Retrieve trained ensemble models (a list of models)
    models = optimizer.result()
    
    # ----- Evaluate on MNIST Test Data -----
    # Aggregate predictions on MNIST test set
    test_samples = dataset.test_size
    agg_preds = np.zeros((test_samples, output_dim))
    x_test, y_true = next(iter(dataset.test_data.batch(test_samples)))
    
    for m in models:
        preds = m.predict(x_test)
        agg_preds += preds
    agg_preds /= len(models)  # Average ensemble predictions

    # Convert predictions to class labels
    predicted_labels = tf.argmax(agg_preds, axis=1)
    
    # Compute classification metrics on MNIST
    accuracy = skmet.accuracy_score(y_true, predicted_labels) * 100
    recall = skmet.recall_score(y_true, predicted_labels, average="micro") * 100
    precision = skmet.precision_score(y_true, predicted_labels, average="macro") * 100
    f1 = skmet.f1_score(y_true, predicted_labels, average="macro")
    
    # Compute MNIST ROC AUC (one-vs-rest for multi-class classification)
    y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=output_dim)
    roc_auc = skmet.roc_auc_score(y_true_one_hot, agg_preds, multi_class="ovr")
    
    # For OOD detection we will use the ensemble's confidence scores.
    # We use the maximum softmax probability as the confidence score.
    confidence_in = np.max(agg_preds, axis=1)  # For MNIST (in-distribution)
    
    # ----- Evaluate on FashionMNIST as OOD -----
    # Load FashionMNIST test data (using tf.keras.datasets)
    (x_fashion_train, y_fashion_train), (x_fashion_test, y_fashion_test) = tf.keras.datasets.fashion_mnist.load_data()
    # Normalize FashionMNIST images to [0, 1]
    x_fashion_test = x_fashion_test.astype('float32') / 255.0
    # FashionMNIST images have shape (28, 28); same as MNIST
    
    num_fashion_samples = x_fashion_test.shape[0]
    agg_preds_fashion = np.zeros((num_fashion_samples, output_dim))
    
    for m in models:
        preds = m.predict(x_fashion_test)
        agg_preds_fashion += preds
    agg_preds_fashion /= len(models)
    
    # Compute confidence scores for FashionMNIST
    confidence_ood = np.max(agg_preds_fashion, axis=1)
    
    # For OOD detection, we assume that in-distribution samples (MNIST) yield higher confidence.
    # We combine the scores from MNIST (label 1) and FashionMNIST (label 0).
    ood_labels = np.concatenate([np.ones_like(confidence_in), np.zeros_like(confidence_ood)])
    ood_scores = np.concatenate([confidence_in, confidence_ood])
    
    # Compute the AUROC for OOD detection
    ood_auroc = skmet.roc_auc_score(ood_labels, ood_scores)
    
    return accuracy, recall, precision, f1, roc_auc, ood_auroc

if __name__ == "__main__":
    num_runs = 5
    results = {"accuracy": [], "recall": [], "precision": [], "f1": [], "roc_auc": [], "ood_auroc": []}

    for i in range(num_runs):
        print(f"Running experiment {i+1}/{num_runs}...")
        acc, rec, prec, f1, roc_auc, ood_auroc = run_experiment(train_steps=8000)
        results["accuracy"].append(acc)
        results["recall"].append(rec)
        results["precision"].append(prec)
        results["f1"].append(f1)
        results["roc_auc"].append(roc_auc)
        results["ood_auroc"].append(ood_auroc)
    
    # Compute mean and standard deviation for each metric
    for metric in results:
        mean = np.mean(results[metric])
        std = np.std(results[metric])
        print(f"{metric.capitalize()}: {mean:.4f} ± {std:.4f}")
