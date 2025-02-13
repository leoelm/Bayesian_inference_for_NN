from . import Optimizer
from Pyesian.nn import BayesianModel
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

def visualize_data(x, y):
    """
    Plots the given x and y data points.

    Parameters:
    x (tf.Tensor or numpy.ndarray): Input values
    y (tf.Tensor or numpy.ndarray): Output values
    """
    # Convert tensors to NumPy arrays if necessary
    x_np = x.numpy().flatten() if isinstance(x, tf.Tensor) else x.flatten()
    y_np = y.numpy().flatten() if isinstance(y, tf.Tensor) else y.flatten()

    # Plot the dataset
    plt.figure(figsize=(8, 6))
    plt.scatter(x_np, y_np, color='blue', alpha=0.5, label="Data Points")
    
    # Plot the best-fit line if data follows a linear trend
    plt.plot(sorted(x_np), sorted(2*x+2), color='red', linewidth=2, label="Best Fit Line")
    
    plt.xlabel("x values")
    plt.ylabel("y values")
    plt.title("Data Visualization")
    plt.legend()
    plt.grid(True)
    plt.show()

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
        self._step = 0
        self._feature_dim = None
        self._k = None
        self._base_model = None
        self._prior = None
        self._posterior_means = []
        self._posterior_stds = []
        self._weights = None

    def step(self, save_document_path=None):
        self._step += 1
        print(f"\nStep {self._step} ------------------------")

        # REQUIRE
        # D: self._dataset
        # q: self._base_model
        # p: self._gp_log_likelihood (HOW DO I ACTUALLY DO THIS?)
        # lambda: self._lambda
        # c: self._generate_measurement_set()[1]

        # 2 ------------------------
        X_D, X_M = self._generate_measurement_set()
        X_M = tf.cast(X_M, dtype=tf.float32)
        X_D = tf.cast(X_D, dtype=tf.float32)
        # ------------------------

        total_dll = None
        total_dkl = None
        
        total_loss = 0

        agg_preds = None
        for i in range(self._k):
            # 3 ------------------------
            self._pack_weights(self._weights[i], self._base_model)
            batch_X, batch_Y = X_D
            noise = tf.random.normal((1, batch_X.shape[1]), mean=0.0, stddev=0.1)
            batch_X += noise
            X_M += noise
            with tf.GradientTape(persistent=True) as tape:
                y_pred_d = self._base_model(batch_X, training=True)
            # ------------------------

            # 4 ------------------------
                log_likelihood = self._dataset.loss()(y_pred_d, batch_Y)
            dll = tape.gradient(log_likelihood, self._base_model.trainable_variables)
            dll = tf.concat([tf.reshape(grad, [-1]) for grad in dll], axis=0)
            dll = (1/self._batch_size * self._k) * dll
            if total_dll is None:
                total_dll = dll
            else:
                total_dll += dll
            # ------------------------

            # 5 ------------------------
            full_samples = tf.concat([batch_X, X_M], axis=0)
            with tf.GradientTape() as tape:
                y_pred = self._base_model(full_samples, training=True)
                gp_ll = self._gp_log_likelihood(y_pred, full_samples)
            
            c1 = tape.gradient(gp_ll, self._base_model.trainable_variables) # red part
            c1 = tf.concat([tf.reshape(grad, [-1]) for grad in c1], axis=0) # red part flattened
            
            c2 = tf.squeeze(self._stein_gradient(self._weights[i]), axis=1) # blue part
            dkl = tf.cast(c2, dtype=tf.float32) - tf.cast(c1, dtype=tf.float32)
            dkl = (1/self._k) * dkl
            if total_dkl is None:
                total_dkl = dkl
            else:
                total_dkl += dkl
            # ------------------------

            total_loss += log_likelihood/self._k

            # print(self.unflatten_gradients(dll - dkl, self._base_model))
            # self.optimizers[i].apply_gradients(zip(self.unflatten_gradients((dll - dkl), self._base_model), self._base_model.trainable_variables))
            # self._weights[i] = self._unpack_weights()
        
            if agg_preds is None:
                agg_preds = y_pred_d
            else:
                agg_preds += y_pred_d

        # 6 ------------------------
        self._pack_weights(self._posterior_means, self._base_model)
        self.optimizers[0].apply_gradients(zip(self.unflatten_gradients((total_dll - total_dkl), self._base_model), self._base_model.trainable_variables))
        self._posterior_means = self._unpack_weights()
        self._update_posterior_parameters()
        # ------------------------

        if self._step % 250 == 0:
            agg_preds /= self._k
            visualize_data(X_D[0], agg_preds)
            # plot_decision_boundary(samples, agg_preds)
            pass

        return total_loss

    def _gp_log_likelihood(self, f, x, lengthscale=1.0, variance=1.0, jitter=1e-6):

        kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=tf.sqrt(variance), length_scale=lengthscale)
    
        # Compute kernel matrix K(x, x)
        K = kernel.matrix(x, x)
        
        # Add jitter for numerical stability
        K += jitter * tf.eye(tf.shape(x)[0], dtype=K.dtype)
        
        # Define the GP prior as a multivariate normal with mean zero
        gp_prior = tfp.distributions.MultivariateNormalFullCovariance(loc=tf.zeros_like(f[:, 0]), covariance_matrix=K)
        
        # Compute log likelihood of the function values under the GP prior
        log_likelihood = gp_prior.log_prob(tf.squeeze(f, axis=-1))
        
        return log_likelihood

        # n = tf.shape(f)[0]
        # K = self._rbf_kernel(f) + noise * tf.eye(n, dtype=f.dtype)
        # L = tf.linalg.cholesky(K)
        # alpha = tf.linalg.cholesky_solve(L, f)
        # log_det_K = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        # ll = -0.5 * tf.squeeze(tf.matmul(tf.transpose(f), alpha)) \
        #     - 0.5 * log_det_K \
        #     - 0.5 * tf.cast(n, f.dtype) * tf.math.log(2 * np.pi)
        # return ll

    def _rbf_kernel(self, x, sigma=1.0):
        diff = tf.expand_dims(x, axis=1) - tf.expand_dims(x, axis=0)
        dist_sq = tf.reduce_sum(tf.square(diff), axis=-1)
        return tf.exp(-dist_sq / (2.0 * sigma**2))
    
    def _grad_rbf_kernel(self, x, sigma=1.0):
        diff = tf.expand_dims(x, axis=1) - tf.expand_dims(x, axis=0)
        kxy = self._rbf_kernel(x, sigma)
        return -diff / (sigma**2) * tf.expand_dims(kxy, axis=-1)

    def _stein_gradient(self, x, sigma=1, reg=1e-3):
        n = x.shape[0]
        x = tf.expand_dims(x, axis=-1)
        K = self._rbf_kernel(x, sigma)
        K_reg = K + reg * tf.eye(n, dtype=x.dtype)
        grad_K = self._grad_rbf_kernel(x, sigma)
        sum_grad_K = tf.reduce_sum(grad_K, axis=1)
        b = (1.0 / n) * sum_grad_K
        return -tf.linalg.solve(K_reg, b)
        
    def _generate_measurement_set(self):
        X_Ds = next(self._data_iterator, (None,None))
        training_data_np = np.asarray(list(self._training_dataset.map(lambda x, y: x)))
        feature_min = training_data_np.min(axis=0)
        feature_max = training_data_np.max(axis=0)
        
        self._feature_dim = training_data_np.shape[1]

        X_M_np = np.random.uniform(low=feature_min, high=feature_max, size=(self._batch_size, training_data_np.shape[1]))
        X_M = tf.convert_to_tensor(X_M_np, dtype=tf.float64)

        if X_Ds[0] is None:
            self._data_iterator = iter(self._dataloader)
            X_Ds = next(self._data_iterator, (None, None))

        return X_Ds, X_M
    
    def unflatten_gradients(self, flat_gradients, model):
        """Unflattens a vector of gradients into a list matching model.trainable_variables."""
        grads = []
        start = 0
        for var in model.trainable_variables:
            # Number of elements in this variable.
            var_shape = var.shape
            num_elements = tf.reduce_prod(var_shape)
            # Extract the corresponding slice from the flat gradients and reshape.
            grad_slice = tf.reshape(flat_gradients[start:start + num_elements], var_shape)
            grads.append(grad_slice)
            start += num_elements
        return grads

    def _update_posterior_parameters(self):
        for row in range(self._weights.shape[0]):
            for col in range(self._weights.shape[1]):
                self._weights[row, col] = tfp.distributions.Normal(self._posterior_means[col], 1).sample()

    def _init_weights(self):
        num_weights = self._get_number_of_trainable_parameters()
        self._weights = np.zeros((self._k, num_weights))
        priors = self._prior.get_model_priors(self._base_model)
        for i in range(self._k):
            trainable_weights = np.array([])
            for layer in priors:
                for val in layer:
                    weights = val.sample()
                    self._posterior_means.append(val.mean().numpy().flatten()[0])
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
        self._lambda = 1/self._batch_size
        self.optimizers = [tf.keras.optimizers.legacy.Adam(learning_rate=self._hyperparameters.lr) for _ in range(self._k)]
        self._init_weights()

        if kwargs["function_prior"] == "GP":
            training_data_np = np.asarray(list(self._training_dataset.map(lambda x, y: x)))
            dataset =tf.convert_to_tensor(training_data_np)
            self._function_prior = tfp.distributions.GaussianProcess(tfp.math.psd_kernels.ExponentiatedQuadratic())

    def update_parameters_step(self):
        pass

    def result(self) -> BayesianModel:
        ensemble = [tf.keras.models.model_from_json(self._model_config) for _ in range(self._k)]
        for i in range(self._k):
            model = ensemble[i]
            self._pack_weights(self._weights[i], model)
        return ensemble

