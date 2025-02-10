from . import Optimizer
import tensorflow as tf
import numpy as np
from tqdm import tqdm

class SVGD(Optimizer):
    def __init__(self):
        super().__init__()
        self._step = 0
        self._M = None
        self._particles = None

    def step(self, save_document_path=None):
        self._step += 1

        samples, labels = next(self._data_iterator, (None,None))
        if samples is None:
            self._data_iterator = iter(self._dataloader)
            samples, labels = next(self._data_iterator, (None, None))
        labels = tf.cast(labels, tf.float32)

        total_loss = 0
        for i in range(self._M):
            updated_particles = np.copy(self._particles[i])
            particles = tf.convert_to_tensor(self._particles[i])
            self._pack_weights(particles, self._base_model)
            k = self._rbf_kernel(particles, particles, self._M)
            print(i, particles.shape)
            for j, particle in tqdm(enumerate(particles)):
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(particles)
                    k = self._rbf_kernel(particles, particle, self._M)
                    predictions = self._base_model(samples)
                    log_likelihood = -self._dataset.loss()(labels, predictions)
                    log_prob_prior = self._calculate_log_prob_particles(particles)
                    
                dk = tf.cast(tape.gradient(k, particles), tf.float32)
                dll = tape.gradient(log_likelihood, self._base_model.trainable_variables)
                dll = tf.concat([tf.reshape(grad, [-1]) for grad in dll], axis=0)
                dlpp = tf.cast(tape.gradient(log_prob_prior, particles), tf.float32)
                
                total_loss += log_likelihood/(samples.shape[0] * self._M * particles.shape[0])

                phi = (1/self._M) * tf.reduce_sum(k * (dll + dlpp) + dk)
                updated_particles[j] += self._hyperparameters.lr * phi
            self._particles[i] = updated_particles
                
        return -1*total_loss

    def _init_particles(self):
        self._particles = np.zeros((self._M, self._num_particles))
        priors = self._prior.get_model_priors(self._base_model)
        for i in range(self._M):
            trainable_weights = np.array([])
            for layer in priors:
                if not layer: # skip any layers without parameter i.e. Flatten
                    continue
                for val in layer:
                    weights = val.sample()
                    trainable_weights = np.concatenate((trainable_weights, weights.numpy().flatten()))

            self._particles[i, :] = trainable_weights

            samples = np.asarray(list(self._training_dataset.map(lambda x, y: x)))
            self._num_datapoints = samples.shape[0]

    def _unpack_weights(self):
        return np.array([x for v in self._base_model.trainable_variables for x in v.numpy().flatten()])

    def _get_number_of_trainable_parameters(self):
        return len([x for v in self._base_model.trainable_variables for x in v.numpy().flatten()])
    
    def _calculate_log_likelihood(self, prediction: tf.Tensor, labels: tf.Tensor, sigma=1.0):
        n = prediction.shape[0]
        return -(n*tf.math.log(2*np.pi))/2 - (n*tf.math.log(sigma**2))/2 - (1/(2*sigma**2))*tf.reduce_sum(tf.square(prediction - labels))
    
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

    def _rbf_kernel(self, a, b, h):
        return tf.cast(tf.exp(-(1/h) * (tf.pow(a-b, 2))), tf.float32)

    def compile_extra_components(self, **kwargs):
        self._batch_size = int(self._hyperparameters.batch_size)
        self._dataset_setup()
        self._base_model = tf.keras.models.model_from_json(self._model_config)
        self._prior = kwargs["prior"]
        self._M = self._hyperparameters.M
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
        ensemble = []
        for i in range(self._M):
            model = tf.keras.models.model_from_json(self._model_config)
            self._pack_weights(self._particles[i], model)
            ensemble.append(model)
        return ensemble

        
