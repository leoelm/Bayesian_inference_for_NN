import sklearn.metrics as met
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import skimage as sk
import os

# //////// Corruptions ////////
def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255

def speckle_noise(x, severity=1):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255

def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]

    x = sk.filters.gaussian(np.array(x) / 255., sigma=c, channel_axis=-1)
    return np.clip(x, 0, 1) * 255

def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255

def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(x) / 255.

    if len(x.shape) < 3 or x.shape[2] < 3:
        x = np.clip(x + c, 0, 1)
    else:
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255

def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

    x = np.array(x) / 255.

    gray_scale = False
    if len(x.shape) < 3 or x.shape[2] < 3:
        x = np.array([x, x, x]).transpose((1, 2, 0))
        gray_scale = True
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)
    if gray_scale:
        x = x[:, :, 0]

    return np.clip(x, 0, 1) * 255

def pixelate(x, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

    x_shape = np.array(x).shape

    x = x.resize((int(x_shape[1] * c), int(x_shape[0] * c)), Image.BOX)

    x = x.resize((x_shape[1], x_shape[0]), Image.NEAREST)

    return x

class Robustness():
    """
        a class representing the robustness analysis of a model
    """
    def __init__(self, model, dataset):
        self.model = model
        self.corruptions = (gaussian_noise, shot_noise, impulse_noise, speckle_noise,
                            gaussian_blur, contrast, brightness, saturate, pixelate)
        self.corruption_dict = {corr_func.__name__: corr_func for corr_func in self.corruptions}
        self.error_dict = {c: None for c in self.corruption_dict.keys()}
        self.n = len(self.corruptions)
        self.severities = np.arange(1, 6)
        self.dataset = dataset
        self.x, self.y_true = next(iter(self.dataset.valid_data.batch(self.dataset.valid_data.cardinality())))
    
    # def c_robustness(self, nb_samples=100):
    #     """
    #     outputs visualisaitons for the corruption robustness analysis. Applies some corruptions to datasets, and outputs accuracy and error rates on predictions.

    #     Args:
    #         dataset (Dataset): dataset to perform analysis on
    #         nb_samples (int): number of samples
    #     """
    #     x, y_true = next(iter(self.dataset.valid_data.batch(self.dataset.valid_data.cardinality())))
    #     _, y_pred = self.model.predict(x, nb_samples)
    #     if self.dataset.likelihood_model == "Classification":
    #         accuracy = met.accuracy_score(y_true, tf.argmax(y_pred, axis = 1))
    #         e_clean = 1 - accuracy
    #         e_s_c = np.array([np.array([self._error_rate(x, y_true, s, c, nb_samples) for s in self.severities]) for c in self.corruptions])
    #         #e_s_c = np.array([np.array([0.8616, 0.876, 0.8796, 0.887, 0.9002]), 
    #         #                  np.array([0.8418, 0.8418, 0.8422000000000001, 0.8398, 0.835]),
    #         #                  np.array([0.852, 0.8588, 0.8626, 0.8704000000000001, 0.8646]),
    #         #                  np.array([0.849, 0.8662, 0.8882, 0.8986, 0.8986])])
    #         ce = np.array([np.sum(e_s_c[i]) for i in range(self.n)])
    #         mce = np.mean(ce)
    #         relative_ce = np.array([np.sum(e_s_c[i] - e_clean) / (self.baseline[i] - self.baseline_clean) * 5 for i in range(self.n)])
    #         mrce = np.mean(relative_ce)
    #         print("Mean Corruption Error", mce*100, "%")
    #         print("Mean Relative Error", mrce*100, "%")
    #         self.plot_ce_by_corruption(ce)
    #         for c in self.corruptions:
    #             self.plot_ce_by_severity(c, e_s_c)
                
    def mean_corruption_error(self, relative=False, nb_samples=100, save_path=None):
        ce = np.array([self.corruption_error(c, helper=True, relative=relative) for c in self.corruption_dict.keys()])
        mean = np.mean(ce)
        if save_path:
            name = "mean_relative_error" if relative else "mean_corruption_error"
            self._save_data(save_path, name, mean)
        else:
            name = "Mean Relative Error:" if relative else "Mean Corruption Error:"
            print(name, mean)
    
    def corruption_error(self, corruption, relative=False, nb_samples=100, save_path=None, helper=False):
        if self.error_dict[corruption] :
            error = self.error_dict[corruption]
        else:
            error = [np.array([self._error_rate(s, corruption, nb_samples) for s in self.severities])]
            self.error_dict[corruption] = error
        if relative:
            _, y_pred = self.model.predict(self.x, nb_samples)
            clean_error = 1 - met.accuracy_score(self.y_true, tf.argmax(y_pred, axis = 1))
            ce = np.sum([x-clean_error for x in error]) / len(self.severities) * 100
        else:
            ce = np.sum(error) / len(self.severities) * 100
        if helper:
            return ce
        if save_path:
            name_file = "relative_error_" if relative else "corruption_error_"
            self._save_data(save_path, name_file + corruption, ce)
        else:
            print_stat = "Relative Error for {}: {}%" if relative else "Corruption Error for {}: {}%"
            print(print_stat.format(corruption, ce))
       
    def robustness_by_corruption(self, nb_samples=100, save_path=None):
        ce = np.array([self.corruption_error(c, nb_samples=nb_samples, helper=True) for c in self.corruption_dict.keys()])
        plt.bar(self.corruptions_dict.keys(), ce)
        plt.xlabel("Corruption")
        plt.ylabel("Corruption Error (%)")
        plt.title("Corruption Error by Corruption")
        self._save_figure(save_path, "robustness_by_corruption") if save_path else plt.show()
        
    def corruption_robustness_by_severity(self, corruption, nb_samples=100, save_path=None):
        if self.error_dict[corruption] :
            error = self.error_dict[corruption]
        else:
            error = [np.array([self._error_rate(s, corruption, nb_samples) for s in self.severities])]
            self.error_dict[corruption] = error
        plt.plot(self.severities, error*100)
        plt.xlabel("Severity")
        plt.ylabel("Error Rate (%)")
        plt.title("Error Rate by Severity for {}".format(corruption))
        self._save_figure(save_path, "corruption_robustness_by_severity") if save_path else plt.show()
    
    def _error_rate(self, severity, corruption, nb_samples):
        images = self.x.numpy()
        #_, width, length, _ = np.shape(x_array)
        #resized = [cv2.resize(image, (224, 224)) for image in x_array]
        corrupted_images = [self._corrupt(image, severity, corruption) for image in images]
        #corrupted_images = [cv2.resize(image, (width, length)) for image in corrupted_images_resized]

        corrupted_inputs = tf.convert_to_tensor(corrupted_images)
        corrupted_inputs = tf.concat(corrupted_inputs, axis=1)
        _, c_predicted = self.model.predict(corrupted_inputs, nb_samples)
        accuracy = met.accuracy_score(self.y_true, tf.argmax(c_predicted, axis = 1))
        error = 1 - accuracy
        #print("Error rate for", corruption, ", with severity", severity, ":", error)
        return error
    
    
    # Image width and height must be at least 32 pixels
    def _corrupt(self, image, severity, corruption):
        if image.ndim == 2:
            image = np.stack((image,)*3, axis=-1) 
        
        height, width, channels = image.shape
        
        if channels == 1:
            image = np.stack((np.squeeze(image),)*3, axis=-1)
        
        image_corrupted = self.corruption_dict[corruption](Image.fromarray(image), severity)
        
        return np.uint8(image_corrupted)
    
    def _save_figure(self, path, name):
        directory = path + "/report/robustness"
        figures = directory + "/figures"
        os.makedirs(directory, exist_ok=True)
        os.makedirs(figures, exist_ok=True)
        plt.savefig(figures + "/" + name + ".png")
        
    def _save_data(self, path, name, content):
        directory = path + "/report/robustness"
        os.makedirs(directory, exist_ok=True)
        file_path = directory + "/" + name + ".txt"
        f = open(file_path, "w")
        f.write(str(content))
        f.close()
    
    