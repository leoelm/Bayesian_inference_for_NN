from . import Optimizer
from Pyesian.nn import BayesianModel
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class FSVI(Optimizer):
    """
    FSVI is a class that inherits from Optimizer.\n
    This inference method is taken from the paper : "Tractable Function-Space Variational Inference in Bayesian Neural Networks". \n
    https://arxiv.org/pdf/2312.17199. \n
    This inference method takes the following hyperparameters:
    Hyperparameters:
        `batch_size`: the size of the batch for one step. Defaults to 128 \n
        `lr`: the learning rate \n
        `pi`: A weight to average between the first and the second prior (only if we have a single prior for the network).
            If we have a single prior this hyperparameter is ignored. 
            This value should be between 0 and 1. \n
        `alpha`: the scale of the KL divergence in the loss function. It should be between 0 and 1 \n
    """

    def __init__(self):
        super().__init__()
        self._feature_dim = None
        self._k = None
        self._base_model = None
        self._prior = None
        self._posterior_means = []
        self._posterior_stds = []
        self._weights = None

    def step(self, save_document_path=None):
        X_D, X_M = self._generate_measurement_set()
        X_M = tf.cast(X_M, dtype=tf.float32)
        X_D = tf.cast(X_D, dtype=tf.float32)
        total_dll = None
        total_dkl = None

        function_prior = tfp.distributions.GaussianProcessRegressionModel(kernel=tfp.math.psd_kernels.ExponentiatedQuadratic(), observation_index_points=X_D[0], observations=X_D[1], index_points=X_M)
        for i in range(self._k):
            self._pack_weights(self._weights[i], self._base_model)
            batch_X, batch_Y = X_D

            # get Log Likelihood Loss
            with tf.GradientTape(persistent=True) as tape:
                y_pred_d = self._base_model(batch_X, training=True)
                log_likelihood = self._dataset.loss()(y_pred_d, batch_Y)

                variational_mean = tf.cast(tf.math.reduce_mean(y_pred_d), dtype=tf.float32)
                variational_cov = tf.cast(tfp.stats.covariance(y_pred_d), dtype=tf.float32)
                variational_dist = tfp.distributions.Normal(variational_mean, variational_cov)

                kl_estimate = 0
                preds = self._base_model(X_M)
                print(tf.transpose(preds).shape)
                log_p_all = function_prior.log_prob(tf.transpose(preds))
                for sample, log_p in zip(X_M, log_p_all):
                    sample = tf.cast(sample, tf.float32)
                    pred = self._base_model(sample)
                    log_q = variational_dist.log_prob(pred)
                    # log_p = function_prior.log_prob(pred)
                    print(log_p)

                    q = variational_dist.prob(pred)

                    kl_estimate += q * (log_q - log_p)
                kl_estimate *= self._lambda


            dll = tape.gradient(log_likelihood, self._base_model.trainable_variables)
            dll = tf.concat([tf.reshape(grad, [-1]) for grad in dll], axis=0)
            dll = (1/self._batch_size) * dll

            if total_dll is None:
                total_dll = dll
            else:
                total_dll += dll

            dkl = tape.gradient(kl_estimate, self._base_model.trainable_variables)
            dkl = tf.concat([tf.reshape(grad, [-1]) for grad in dkl], axis=0)

            if total_dkl is None:
                total_dkl = dkl
            else:
                total_dkl += dkl

        print(total_dkl)
        
        # predictions = []
        # for i in range(self._k):
        #     self._pack_weights(self._weights[i], self._base_model)
        #     predictions.append(self._base_model(X_D).numpy().flatten())

        # # get KL gradient
        # variational_mean = tf.cast(tf.math.reduce_mean(predictions, axis=0), dtype=tf.float32)
        # variational_cov = tf.cast(tfp.stats.covariance(predictions, sample_axis=0), dtype=tf.float32)
        # prior_mean = tf.cast(function_prior.mean(), dtype=tf.float32)
        # prior_cov = tf.cast(function_prior.covariance(), dtype=tf.float32)

        # print(variational_mean, prior_mean)
        # print(variational_cov, prior_cov)

        # variational_gaussian = tfp.distributions.MultivariateNormalDiag(variational_mean, variational_cov)
        # prior_gaussian = tfp.distributions.MultivariateNormalDiag(prior_mean, prior_cov[0])

        # print(variational_gaussian.event_shape)
        # print(prior_gaussian.event_shape)

        # kl_divergence = tfp.distributions.kl_divergence(variational_gaussian, prior_gaussian)
        # print(kl_divergence)
        # with tf.GradientTape(persistent=True) as tape:
        #     kl_divergence = tfp.distributions.kl_divergence(variational_gaussian, prior_gaussian)

        # dkl = tape.gradient(kl_divergence, self._base_model.trainable_variables)

        # if not total_dkl:
        #     total_dkl = dkl
        # else:
        #     total_dkl += dkl

        
        print(total_dll)
        print(total_dkl * self._lambda)
            

    def _generate_measurement_set(self):
        X_Ds = next(self._data_iterator, (None,None))
        training_data_np = np.asarray(list(self._training_dataset.map(lambda x, y: x)))
        feature_min = training_data_np.min(axis=0)
        feature_max = training_data_np.max(axis=0)
        
        self._feature_dim = training_data_np.shape[1]

        X_M_np = np.random.uniform(low=feature_min, high=feature_max, size=(self._batch_size, training_data_np.shape[1]))
        X_M = tf.convert_to_tensor(X_M_np, dtype=tf.float64)

        if X_Ds is None:
            self._data_iterator = iter(self._dataloader)
            X_Ds, _ = next(self._data_iterator, (None, None))

        return X_Ds, X_M

    def _init_weights(self):
        num_weights = self._get_number_of_trainable_parameters()
        self._weights = np.zeros((self._k, num_weights))
        priors = self._prior.get_model_priors(self._base_model)
        for i in range(self._k):
            trainable_weights = np.array([])
            for layer in priors:
                for val in layer:
                    weights = val.sample()
                    trainable_weights = np.concatenate((trainable_weights, weights.numpy().flatten()))

            self._weights[i, :] = trainable_weights

            samples = np.asarray(list(self._training_dataset.map(lambda x, y: x)))
            self._num_datapoints = samples.shape[0]

    def _unpack_weights(self):
        return np.array([x for v in self._base_model.trainable_variables for x in v.numpy().flatten()])
    
    def _pack_weights(self, weights: tf.Tensor, model):
        curr = 0
        weights = tf.cast(weights, tf.float32)
        for layer_idx, layer in enumerate(model.layers):
            for param_idx, params in enumerate(layer.trainable_variables):
                a, b = params.shape[0], params.shape[1] if len(params.shape) > 1 else 1
                end = a * b + curr
                vals = tf.reshape(weights[curr:end], params.shape)
                curr = end
                model.layers[layer_idx].trainable_variables[param_idx].assign(vals)

    def _get_number_of_trainable_parameters(self):
        return len([x for v in self._base_model.trainable_variables for x in v.numpy().flatten()])

    def compile_extra_components(self, **kwargs):
        self._batch_size = int(self._hyperparameters.batch_size)
        self._dataset_setup()
        self._k = self._hyperparameters.k
        self._base_model = tf.keras.models.model_from_json(self._model_config)
        self._prior = kwargs["prior"]
        self._lambda = self._hyperparameters._lambda
        self._init_weights()

        if kwargs["function_prior"] == "GP":
            training_data_np = np.asarray(list(self._training_dataset.map(lambda x, y: x)))
            dataset =tf.convert_to_tensor(training_data_np)
            self._function_prior = tfp.distributions.GaussianProcess(tfp.math.psd_kernels.ExponentiatedQuadratic())

    def update_parameters_step(self):
        pass

    def result(self) -> BayesianModel:
        pass

