from . import Optimizer
from Pyesian.nn import BayesianModel
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class SpectralSteinEstimator:
    def __init__(self):
        pass

    def _rbf_kernel(self, x, y, length_scale=1, variance=1):
        dist = tf.reduce_sum(x**2, axis=1, keepdims=True) - 2 * tf.matmul(x, y, transpose_b=True) + tf.reduce_sum(y**2, axis=1)
        return variance * tf.exp(-dist / (2 * length_scale**2))
    
    def _compute_stein_kernel(self, x, y, score_func, sigma):
        with tf.GradientTape() as tape:
            tape.watch(x)
            k = self._rbf_kernel(x, y, sigma)

            score = score_func(x)
            grad_k = tape.gradient(k, x)[0]
            print(grad_k.shape)
            print(score.shape)
            print(k.shape)
            print(tf.matmul(k, score).shape)
            stein_mat = grad_k + tf.matmul(k, score)

        return stein_mat, k
    
    def _extract_eigenfunctions(stein_mat, num_eigen=5):
        vals, vecs = tf.linalg.eigh(stein_mat)
        return vals[:num_eigen], vecs[:, :num_eigen]
    
    def compute_gradients(self, samples, bnn_output, score_func, prior, num_eigen=5, sigma=1):

        prior_mean = prior.mean()
        prior_cov = self._rbf_kernel(samples, samples, sigma)
        score_prior = score_func(bnn_output, prior_mean, prior_cov)

        stein_mat, _ = self._compute_stein_kernel(samples, samples, lambda x: score_prior, sigma)
        eigenvalues, eigenfunctions = self._extract_eigenfunctions(stein_mat, num_eigen=num_eigen)
        
        gradient = 0
        for val, vec in zip(eigenvalues, eigenfunctions):
            weight = 1.0 / val
            gradient += weight * vec

        return gradient
        

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

    def _sample_model_weights(self):
        for layer_idx in range(len(self._base_model.layers)):
            layer = self._base_model.layers[layer_idx]
            if len(layer.trainable_variables) != 0:
                for i in range(len(layer.trainable_variables)):
                    weight = tfp.distributions.Normal(self._posterior_means[layer_idx][i], self._posterior_stds[layer_idx][i]).sample()
                    self._base_model.layers[layer_idx].trainable_variables[i].assign(weight)

    def _calculate_log_likelihood(self, prediction: tf.Tensor, labels: tf.Tensor, sigma=1.0):
        n = prediction.shape[0]
        return -(n*tf.math.log(2*np.pi))/2 - (n*tf.math.log(sigma**2))/2 - (1/(2*sigma**2))*tf.reduce_sum(tf.square(prediction - labels))

    def _sample_gp_prior(self, index_points, lengthscale=1, variance=1, jitter=1e-6):
        K = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale = lengthscale, amplitude = variance).apply(index_points, index_points)
        num_points = tf.shape(index_points)[0]
        K += jitter * tf.eye(num_points)
        L = tf.linalg.cholesky(K)
        z = tf.random.normal(shape=(num_points, 1))
        return tf.matmul(L, z)
    
    def _kde_estimate(self, f, f_samples, bandwidth=0.1):
        f_flat = tf.reshape(f, [-1])
        M = tf.shape(f_samples)[0]
        densities = []

        for i in range(M):
            f_i = tf.reshape(f_samples[i], [-1])
            diff = f_flat - f_i
            exponent = -0.5 * tf.reduce_sum(tf.square(diff)) / (bandwidth ** 2)

            d = tf.cast(tf.size(f_flat), tf.float32)
            norm_const = tf.pow(2.0 * np.pi * (bandwidth ** 2), d / 2)
            densities.append(tf.exp(exponent) / norm_const)
        densities = tf.stack(densities)
        return tf.reduce_mean(densities)
    
    def _compute_log_gp_density(self, f, anchor_points, lengthscale=1.0, variance=1.0, jitter=1e-6):
        K = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale = lengthscale, amplitude = variance).apply(anchor_points, anchor_points)
        n = tf.shape(anchor_points)[0]
        K += jitter * tf.eye(n, dtype=tf.float32)
        
        # Cholesky factorization.
        L = tf.linalg.cholesky(K)
        
        # Solve for alpha: K^{-1} f = L^{-T}(L^{-1} f)
        alpha = tf.linalg.cholesky_solve(L, f)
        
        # Compute log-determinant from the Cholesky factor.
        log_det_K = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        
        n_float = tf.cast(n, tf.float32)
        log_prob = -0.5 * tf.matmul(tf.transpose(f), alpha)
        log_prob -= 0.5 * n_float * tf.math.log(2.0 * np.pi)
        log_prob -= 0.5 * log_det_K
        return tf.squeeze(log_prob)  # scalar
    
    def _estimate_kl(self, x, mc_samples=10):

        f_samples = []
        for _ in range(mc_samples):
            f_samples.append(self._base_model(x))
        
        for i in range(mc_samples):
            f_i = f_samples[i]
            q_f = self._kde_estimate(f_i, f_samples)
            log_p = self._compute_log_gp_density(f_i, x)


    def step(self, save_document_path=None):
        X_D, X_M = self._generate_measurement_set()

        for i in range(self._k):
            self._pack_weights(self._weights[i], self._base_model)
            batch_X, batch_Y = X_D
            y_pred = self._base_model(batch_X, training=True)

            # get Log Likelihood Loss
            log_likelihood = self._calculate_log_likelihood(y_pred, batch_Y)

            kl_estimate = self._estimate_kl(X_M)

            print(log_likelihood)
            print(kl_estimate)

        # X_Ds, X_M = self._generate_measurement_set()

        # total_gradient = None
        # for i in range(self._k):
        #     with tf.GradientTape(persistent=True) as tape:
        #         self._pack_weights(self._weights[i], self._base_model)
        #         features, labels = X_Ds
        #         noise = tfp.distributions.Normal(
        #                     tf.zeros(self._feature_dim),
        #                     tf.ones(self._feature_dim)).sample()
        #         noisy_features = features + tf.cast(noise, tf.float32)

        #         for sample, label in zip(noisy_features, labels):
        #             x = self._base_model(sample, training=True)
        #             gp = tfp.distributions.GaussianProcess(tfp.math.psd_kernels.ExponentiatedQuadratic(), index_points=[x])
        #             log_likelihood = gp.log_prob(tf.cast(label, dtype=tf.float32))

        #             if total_gradient is None:
        #                 total_gradient = np.array([val.numpy().flatten()[0] for val in tape.gradient(log_likelihood, self._base_model.trainable_variables)])
        #             else:
        #                 total_gradient += np.array([val.numpy().flatten()[0] for val in tape.gradient(log_likelihood, self._base_model.trainable_variables)])

        # total_gradient /= self._k
        # total_gradient /= self._batch_size

        # self._kl_estimate(X_M)

    def _calculate_log_prob_weights(self, weights: tf.Tensor):
        curr = 0
        log_prob = 0
        weights = tf.cast(weights, tf.float32)
        for layer in self._prior.get_model_priors(self._base_model):
            for vars in layer:
                end = tf.math.reduce_prod(vars.batch_shape_tensor()) + curr
                layer_weights = tf.reshape(weights[curr:end], vars.batch_shape)
                log_prob += tf.reduce_sum(vars.log_prob(layer_weights))

                curr = end

        return log_prob
            

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
        self._init_weights()

        if kwargs["function_prior"] == "GP":
            training_data_np = np.asarray(list(self._training_dataset.map(lambda x, y: x)))
            dataset =tf.convert_to_tensor(training_data_np)
            self._function_prior = tfp.distributions.GaussianProcess(tfp.math.psd_kernels.ExponentiatedQuadratic(), index_points=dataset)
            pass
        # elif kwargs["function_prior"] == "Normal":
        #     self._function_prior = lambda y_pred: tfp.distributions.Normal(y_pred, 1)
        # pass

    def update_parameters_step(self):
        pass

    def result(self) -> BayesianModel:
        pass

