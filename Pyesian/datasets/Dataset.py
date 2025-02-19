import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import os
from PIL import Image
from ucimlrepo import fetch_ucirepo 


class Dataset:
    """
        a class representing a dataset with its portions, training dataset, validation dataset, testing dataset
        and its loss function
    """
    test_size: int
    valid_size: int
    train_size: int
    train_data: tf.data.Dataset
    test_data: tf.data.Dataset
    valid_data: tf.data.Dataset
    size: int

    def __init__(self,
                 dataset, loss, likelihoodModel="Classification",
                 load_images=False,
                 target_dim=1,
                 feature_normalisation=False, label_normalisation=False,
                 train_proportion=0.8,
                 test_proportion=0.1, valid_proportion=0.1):
        """
        constructor of a dataset

        Args:
            dataset: if the dataset is a string and there exists a tf dataset with the given name,
            we read from tf datasets. Otherwise, you can provide a tf.data.Dataset combining inputs and labels 
            (please use tf.data.Dataset.from_tensor_slices), or you can provide a pd.Dataframe or a path to a csv file.
            loss (tf.keras.losses): The loss function for the dataset
            likelihoodModel (str, optional): the training type, could be "Classification" or "Regression". Defaults to "Classification".
            normalise (bool, optional): nomalise the data or not. Defaults to True.

        Raises:
            ValueError: When the split propostions don't sum up to 1
            ValueError: When an unsupported dataset type is passed. We support the following: \n 
            tf datasets with "image" and "label"\n
            tf.data.Dataset \n
            pd.Dataframe \n
            csv files \n
            folder with subfolder "images" and csv file "labels"
        """
        if(train_proportion + test_proportion + valid_proportion != 1):
            raise ValueError("Dataset split test_proportions must sum up to 1")
        self._train_proportion = train_proportion
        self._test_proportion = test_proportion
        self._valid_proportion = valid_proportion
        self._loss = loss
        self.likelihood_model = likelihoodModel
        self.target_dim = target_dim
        self._label_mean = None
        self._label_std = None
        if isinstance(dataset, str) and (dataset in tfds.list_builders()):
            dataset = tfds.load(dataset, split="train", try_gcs=True)
            dataset = dataset.map(lambda data: [data["image"], data["label"]])
            self._init_from_tf_dataset(dataset)
        elif (isinstance(dataset, tf.data.Dataset)):
            self._init_from_tf_dataset(dataset)
        elif (isinstance(dataset, pd.DataFrame)):
            self._init_from_dataframe(dataset)
        elif (isinstance(dataset, str) and load_images == False):
            self._init_from_csv(dataset)
        elif (isinstance(dataset, str) and load_images == True):
            input, label = self._load_images_and_csv(dataset)
            dataset = tf.data.Dataset.from_tensor_slices((input, label))
            self._init_from_tf_dataset(dataset)
        elif (isinstance(dataset, int)):
            self._init_from_uci(dataset)
        else:
            raise ValueError("Unsupported dataset format")
        self.train_data = self.train_data.cache()
        self.train_data = self.train_data.shuffle(self.train_data.cardinality())
        self.train_data = self.train_data.prefetch(tf.data.AUTOTUNE)

        if (feature_normalisation):
            self.feature_normalisation()
        if (label_normalisation):
            self.label_normalisation()

    def _load_images_from_directory(self, dir):
        images = []
        for imagestr in sorted(os.listdir(dir)):
            if imagestr.endswith('.png') or imagestr.endswith('.jpg'):
                image = Image.open(os.path.join(dir, imagestr)).convert('L')
                images.append(np.asarray(image))
        return np.array(images)

    def _load_images_and_csv(self, directory):
        """
        Return the dataset as numpy arrays.

        Arguments:
            directory (str): path to the dataset directory
        Returns:
            images (array): images of the datasets, of shape (N,H,W)
            labels (array): labels corresponding to images, of shape (N',)
        """
        images = self._load_images_from_directory(os.path.join(directory, 'images'))
        labels = np.loadtxt(os.path.join(directory, 'labels.csv'), dtype=int)
        return images, labels

    def _init_from_tf_dataset(self, dataset: tf.data.Dataset):
        dataset = dataset.shuffle(dataset.cardinality())
        self.size = tf.data.experimental.cardinality(dataset).numpy()
        self.train_size = int(self._train_proportion * self.size)
        self.test_size = int(self._test_proportion * self.size)
        self.valid_size = int(self._valid_proportion * self.size)
        self.train_data = dataset.take(self.train_size)
        self.test_data = dataset.skip(self.train_size)
        self.valid_data = self.test_data.skip(self.test_size)
        self.test_data = self.test_data.take(self.test_size)

    def _init_from_dataframe(self, dataframe: pd.DataFrame):
        features = dataframe.iloc[:, :-self.target_dim]
        targets = dataframe.iloc[:, -self.target_dim:]
        dataset = tf.data.Dataset.from_tensor_slices((features.values, targets.values))
        self._init_from_tf_dataset(dataset)
        return targets.values

    def _init_from_csv(self, filename: str):
        dataframe = pd.read_csv(filename)
        return self._init_from_dataframe(dataframe)
    
    def _init_from_uci(self, id: int):
        dataframe = fetch_ucirepo(id=id)
        features = dataframe.data.features
        targets = dataframe.data.targets
        dataset = tf.data.Dataset.from_tensor_slices((features, targets))
        self._init_from_tf_dataset(dataset)
        return targets

    def training_dataset(self) -> tf.data.Dataset:
        """
        returns the training dataset

        Returns:
            tf.data.Dataset: the training dataset
        """
        return self.train_data

    def loss(self, reduction='auto'):
        """
        returns the loss function to be used on the dataset

        Returns:
            tf.keras.losses: the loss function
        """
        return self._loss(reduction=reduction)

    def input_shape(self):
        """gives the input shape for the dataset

        Returns:
            tuple: The input shape
        """
        return next(iter(self.train_data))[0].shape

    def _normalise_feature(self, x, y, mean, std):
        x = tf.cast(x, dtype=std.dtype)
        y = tf.cast(y, dtype=std.dtype)
        mean = tf.cast(mean, dtype=std.dtype)
        return ((x - mean) / std, y)

    def _normalise_label(self, x, y, mean, std):
        return (x, (y - mean) / std)

    def label_normalisation(self):
        """
            normalises the dataset labels
        """
        if self.likelihood_model == "Regression":
            input, label = next(iter(self.train_data.batch(self.train_data.cardinality().numpy() / 10)))
            self._label_mean = tf.reduce_mean(label)
            self._label_std = tf.math.reduce_std(tf.cast(label, dtype=tf.float64))
            self.train_data = self.train_data.map(
                lambda x, y: self._normalise_label(x, y, self._label_mean, self._label_std + 1e-8),
                num_parallel_calls=tf.data.AUTOTUNE)
            self.valid_data = self.valid_data.map(
                lambda x, y: self._normalise_label(x, y, self._label_mean, self._label_std + 1e-8),
                num_parallel_calls=tf.data.AUTOTUNE)
            self.test_data = self.test_data.map(
                lambda x, y: self._normalise_label(x, y, self._label_mean, self._label_std + 1e-8),
                num_parallel_calls=tf.data.AUTOTUNE)

    def feature_normalisation(self):
        """
            normalises the dataset features
        """
        if self.likelihood_model == "Regression":
            input, label = next(iter(self.train_data.batch(self.train_data.cardinality().numpy())))
            mean = tf.reduce_mean(input, axis=0)
            std = tf.math.reduce_std(tf.cast(input, dtype=tf.float64), axis=0)
            self.train_data = self.train_data.map(lambda x, y: self._normalise_feature(x, y, mean, std + 1e-8),
                                                  num_parallel_calls=tf.data.AUTOTUNE)
            self.valid_data = self.valid_data.map(lambda x, y: self._normalise_feature(x, y, mean, std + 1e-8),
                                                  num_parallel_calls=tf.data.AUTOTUNE)
            self.test_data = self.test_data.map(lambda x, y: self._normalise_feature(x, y, mean, std + 1e-8),
                                                num_parallel_calls=tf.data.AUTOTUNE)
        else:
            self.train_data = self.train_data.map(lambda x, y: (tf.math.divide(tf.cast(x,tf.float32), 255), y),
                                                  num_parallel_calls=tf.data.AUTOTUNE)
            self.valid_data = self.valid_data.map(lambda x, y: (tf.math.divide(tf.cast(x,tf.float32), 255), y),
                                                  num_parallel_calls=tf.data.AUTOTUNE)
            self.test_data = self.test_data.map(lambda x, y: (tf.math.divide(tf.cast(x,tf.float32), 255), y),
                                                num_parallel_calls=tf.data.AUTOTUNE)
