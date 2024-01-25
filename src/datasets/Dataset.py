from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd




"""
class Dataset:
    testData: dict
    trainData: dict
    validData: dict
    likelihoodModel: str
    scaler: MinMaxScaler

    def __init__(self, train=None, test=None, valid=None, likelihood=None):
        self.testData = test
        self.trainData = train
        self.validData = valid
        self.likelihoodModel = likelihood
        self.scaler = MinMaxScaler()

    def normalise(self):
        self.trainData["input"] = self.scaler.fit_transform(self.trainData["input"])
        self.testData["input"] = self.scaler.transform(self.testData["input"])
        self.validData["input"] = self.scaler.transform(self.validData["input"])
"""

class Dataset:
    test_size: int
    valid_size: int
    train_size: int
    train_data: tf.data.Dataset
    test_data: tf.data.Dataset
    valid_data: tf.data.Dataset
    size: int

    def __init__(self, dataset: tf.data.Dataset, size: int):
        self.train_size = int(0.8 * size)
        self.test_size = int(0.1 * size)
        self.valid_size = int(0.1 * size)
        self.train_data = dataset.take(self.train_size)
        self.test_data = dataset.skip(self.train_size)
        self.valid_data = self.test_data.skip(self.test_size)
        self.test_data = self.test_data.take(self.test_size)

    def tf_dataset(self):
        return self.train_data
    
    def normalise(self):
        pass


"""
def convert_sklearn_dataset(dataset, likelihood):
    df = dataset["data"].insert(4, "target", dataset["target"])
    print(df)
    train, test_valid = train_test_split(df, test_size=0.2, shuffle=True)
    test, valid = train_test_split(test_valid, test_size=0.5, shuffle=True)
    train_data = {"input": None, "labels": None}
    test_data = {"input": None, "labels": None}
    valid_data = {"input": None, "labels": None}
    train_data["labels"] = tf.convert_to_tensor(train["target"])
    test_data["labels"] = tf.convert_to_tensor(test["target"])
    valid_data["labels"] = tf.convert_to_tensor(valid["target"])
    train_data["input"] = tf.convert_to_tensor(train.drop("target"))
    test_data["input"] = tf.convert_to_tensor(test.drop("target"))
    valid_data["input"] = tf.convert_to_tensor(valid.drop("target"))
    new_dataset = Dataset(train=train_data, test=test_data, valid=test_data, likelihood=likelihood)
    return new_dataset
"""
"""
def convert_csv_dataset(filename, likelihood):
    df = pd.read_csv(filename)
    train, test_valid = train_test_split(df, test_size=0.2, shuffle=True)
    test, valid = train_test_split(test_valid, test_size=0.5, shuffle=True)
    train_data = {"input": None, "labels": None}
    test_data = {"input": None, "labels": None}
    valid_data = {"input": None, "labels": None}
    train_data["labels"] = tf.convert_to_tensor(train["target"])
    test_data["labels"] = tf.convert_to_tensor(test["target"])
    valid_data["labels"] = tf.convert_to_tensor(valid["target"])
    train_data["input"] = tf.convert_to_tensor(train.drop("target"))
    test_data["input"] = tf.convert_to_tensor(test.drop("target"))
    valid_data["input"] = tf.convert_to_tensor(valid.drop("target"))
    new_dataset = Dataset(train=train_data, test=test_data, valid=test_data, likelihood=likelihood)
    return new_dataset
"""

#pass dataset from 

def load_tf_dataset(name):
    data = tfds.load(name, split='train', shuffle_files=True)
    assert isinstance(data, tf.data.Dataset)
    dataset = convert_tf_dataset(data, tf.data.experimental.cardinality(data).numpy())

def convert_tf_dataset(dataset, size):
    return Dataset(dataset, size)


load_tf_dataset('mnist')
"""
data = tfds.load('mnist', split='train')
assert isinstance(data, tf.data.Dataset)
print(data.take(10))
print(data.__dict__)
dataset = convert_tf_dataset(data, 100)
print(dataset.train_data)"""