from . import Optimizer
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

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

def plot_decision_boundary(x, y):
    y = np.argmax(y, axis=1)
    # Scatter plot with color-coded labels
    plt.figure(figsize=(8, 6))
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("2D Feature Visualization with Labels")
    plt.colorbar(label="Class Label")
    plt.show()

class SVGD(Optimizer):
    def __init__(self):
        super().__init__()
        self._step = 0
        self._M = None
        self._particles = None
    
    def _svgd_gradients(self, particle_idx, log_prob_gradients, kernel):
        tensor_particles = tf.convert_to_tensor(self._particles)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(tensor_particles)
            kernel_matrix = kernel(tensor_particles, tensor_particles)
            kernel_matrix_sum = tf.reduce_sum(kernel_matrix)

        grad_kernel = -tape.gradient(kernel_matrix_sum, tensor_particles)/2
        kernel_matrix = tf.cast(kernel_matrix, tf.float32)
        grad_kernel = tf.cast(grad_kernel, tf.float32)
        log_prob_gradients = tf.repeat(tf.expand_dims(log_prob_gradients, axis=0), repeats=kernel_matrix.shape[1], axis=0)
        weighted_grad = tf.matmul(kernel_matrix, log_prob_gradients)

        phi = (weighted_grad + grad_kernel) / self._M
        return phi
    
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

    def step(self, save_document_path=None):

        def log_prior(theta):
            return tf.reduce_sum(tfp.distributions.Normal(0, 1).log_prob(theta))

        self._step += 1

        samples, labels = next(self._data_iterator, (None,None))
        if samples is None:
            self._data_iterator = iter(self._dataloader)
            samples, labels = next(self._data_iterator, (None, None))
        labels = tf.cast(labels, tf.float32)

        total_loss = 0
        agg_preds = None
        for i in range(self._M):
            particles = tf.cast(tf.convert_to_tensor(self._particles[i]), dtype=tf.float32)
            self._pack_weights(particles, self._base_model)

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(particles)
                y_hat = self._base_model(samples)
                log_likelihood = self._dataset.loss()(labels, y_hat)
                lp = log_prior(particles)
            
            dll = tape.gradient(log_likelihood, self._base_model.trainable_variables)
            flattened_dll = tf.concat([tf.reshape(grad, [-1]) for grad in dll], axis=0)
            dlp = tape.gradient(lp, particles) # something is wrong with this, when not adding, works on simple classification
            
            if agg_preds is None:
                agg_preds = y_hat
            else:
                agg_preds += y_hat

            gradients = self._svgd_gradients(i, flattened_dll, self.rbf_kernel)[i]
            self.optimizers[i].apply_gradients(zip(self.unflatten_gradients(gradients, self._base_model), self._base_model.trainable_variables))
            updated_particles = self._unpack_weights()
            self._particles[i] = updated_particles
            
            total_loss += log_likelihood / self._M
        
        if self._step % 100 == 0:
            agg_preds /= self._M
            # visualize_data(samples, agg_preds)
            # plot_decision_boundary(samples, agg_preds)
            pass

        return total_loss

    def _init_particles(self):
        self._particles = np.zeros((self._M, self._num_particles))
        priors = self._prior.get_model_priors(self._base_model)
        samples = np.asarray(list(self._training_dataset.map(lambda x, y: x)))
        self._num_datapoints = samples.shape[0]
        for i in range(self._M):
            trainable_weights = np.array([])
            for layer in priors:
                if not layer: # skip any layers without parameter i.e. Flatten
                    continue
                for val in layer:
                    weights = val.sample()
                    trainable_weights = np.concatenate((trainable_weights, weights.numpy().flatten()))

            self._particles[i, :] = trainable_weights

    def _unpack_weights(self):
        return np.array([x for v in self._base_model.trainable_variables for x in v.numpy().flatten()])

    def _get_number_of_trainable_parameters(self):
        return len([x for v in self._base_model.trainable_variables for x in v.numpy().flatten()])

    def baseline__kernel(self, h = -1):
        sq_dist = pdist(self._particles)
        pairwise_dists = squareform(sq_dist)**2
        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)  
            h = np.sqrt(0.5 * h / np.log(self._particles.shape[0]+1))

        # compute the rbf kernel
        
        Kxy = np.exp( -pairwise_dists / h**2 / 2)

        dxkxy = -np.matmul(Kxy, self._particles)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(self._particles.shape[1]):
            dxkxy[:, i] = dxkxy[:,i] + np.multiply(self._particles[:,i],sumkxy)
        dxkxy = dxkxy / (h**2)
        return (Kxy, dxkxy)

    def rbf_kernel(self, x, y, gamma=1.0):
        """
        Computes the RBF (Gaussian) kernel between each pair of rows in x and y.
        
        Args:
            x: Tensor of shape (n, d)
            y: Tensor of shape (m, d)
            gamma: Kernel coefficient (default 1.0)
        
        Returns:
            A tensor of shape (n, m) where each entry is exp(-gamma * ||x_i - y_j||^2)
        """
        # Expand dimensions to compute pairwise differences:
        #   x_exp: shape (n, 1, d)
        #   y_exp: shape (1, m, d)
        x_exp = tf.expand_dims(x, axis=1)
        y_exp = tf.expand_dims(y, axis=0)
        diff = x_exp - y_exp  # shape: (n, m, d)
        dist_sq = tf.reduce_sum(tf.square(diff), axis=-1)  # shape: (n, m)
        return tf.exp(-gamma * dist_sq)

    def _calculate_log_prob_particles(self, particles: tf.Tensor):
        curr = 0
        log_prob = 0
        particles = tf.cast(particles, tf.float32)
        for layer in self._prior.get_model_priors(self._base_model):
            if not layer: # skip layers without parameters e.g. Flatten
                continue
            for vars in layer:
                end = tf.math.reduce_prod(vars.batch_shape_tensor()) + curr
                layer_particles = tf.reshape(particles[curr:end], vars.batch_shape)
                log_prob += tf.reduce_sum(vars.log_prob(layer_particles))

                curr = end

        return log_prob
    def compile_extra_components(self, **kwargs):
        self._batch_size = int(self._hyperparameters.batch_size)
        self._dataset_setup()
        self._base_model = tf.keras.models.model_from_json(self._model_config)
        self._prior = kwargs["prior"]
        self._M = self._hyperparameters.M
        self._lr = self._hyperparameters.lr
        self.optimizers = [tf.keras.optimizers.Adam(learning_rate=self._lr) for _ in range(self._M)]
        self._num_particles = self._get_number_of_trainable_parameters()
        self._init_particles()

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

    def update_parameters_step(self):
        pass

    def result(self):
        ensemble = [tf.keras.models.model_from_json(self._model_config) for _ in range(self._M)]
        for i in range(self._M):
            model = ensemble[i]
            self._pack_weights(self._particles[i], model)
        return ensemble

        
