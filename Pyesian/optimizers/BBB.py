import math
import os
from Pyesian.distributions import GaussianPrior
from Pyesian.distributions.tf import TensorflowProbabilityDistribution
from Pyesian.nn import BayesianModel
from . import Optimizer
import tensorflow as tf
import tensorflow_probability as tfp



class BBB(Optimizer):

    """
    BBB is a class that inherits from Optimizer.\n
    This inference method is taken from the paper : "Weight Uncertainty in Neural Networks". \n
    https://arxiv.org/pdf/1505.05424.pdf. \n
    This inference methods takes the following hyperparameters:
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
        self._data_iterator = None
        self._dataloader = None
        self._lr = None
        self._alpha = None
        self._base_model: tf.keras.Model = None
        self._priors_list = None
        self._weight_layers_indices = []
        self._posterior_mean_list = []
        self._posterior_std_dev_list = []
        self._priors_list = []
        self._layers_intervals = [] # saves interval of (n, n) for each layer n that has trainable parameters
        self._prior2: GaussianPrior = None
        self._prior1: GaussianPrior = None
        self._prior: GaussianPrior = None
        self._step = 0

        self.val_losses  = []
        self.train_losses = []



    def _guassian_likelihood(self,weights, mean, std_dev):
        """
        calculates the likelihood of the weights being from a guassian distribution of the given mean and standard deviation

        Args:
            weights (tf.Tensor): the weights to test
            mean (tf.Tensor): the mean
            std_dev (tf.Tensor): the standard deviation

        Returns:
            tf.Tensor: the likelihood
        """
        guassian_distribution = tfp.distributions.Normal(mean, tf.math.softplus(std_dev))
        return tf.reduce_sum(guassian_distribution.log_prob(weights))

    def _prior_guassian_likelihood(self):
        """
        calculates the guassian likelihood of the weights with respect to the prior
        Returns:
            tf.Tensor: the likelihood of the weights of being sampled from the prior
        """
        likelihood = 0
        mean_idx = 0
        for layer_idx in range(len(self._base_model.layers)):
            layer = self._base_model.layers[layer_idx]
            for i in range(len(layer.trainable_variables)):
                likelihood += self._guassian_likelihood(
                    layer.trainable_variables[i], 
                    self._priors_list[mean_idx][i].mean(),
                    self._priors_list[mean_idx][i].stddev(),
                )
            mean_idx += 1

        return likelihood
    
    def _posterior_guassian_likelihood(self):
        """
        calculates the guassian likelihood of the weights with respect to the posterior
        Returns:
            tf.Tensor: the likelihood of the weights of being sampled from the posterior
        """
        likelihood = 0
        mean_idx = 0
        for layer_idx in range(len(self._base_model.layers)):
            layer = self._base_model.layers[layer_idx]
            for i in range(len(layer.trainable_variables)):
                likelihood += self._guassian_likelihood(
                    layer.trainable_variables[i], 
                    self._posterior_mean_list[mean_idx][i], 
                    self._posterior_std_dev_list[mean_idx][i]
                )
            if len(layer.trainable_variables) != 0:
                mean_idx += 1

        return likelihood

    def _cost_function(self, labels: tf.Tensor, predictions: tf.Tensor):
        """
        calculate the cost function as follows:
        cost = data_likelihood + alpha * kl_divergence

        Args:
            labels (tf.Tensor): the labels
            predictions (tf.Tensor): the predictions

        Returns:
            tf.Tensor: the cost
        """
        posterior_likelihood = self._posterior_guassian_likelihood()
        prior_likelihood = self._prior_guassian_likelihood()
        kl_divergence = tf.math.subtract(posterior_likelihood, prior_likelihood)
        data_likelihood = self._dataset.loss()(labels, predictions)
        kl_divergence = tf.multiply(kl_divergence, self._alpha)
        return tf.math.add(data_likelihood, kl_divergence)
    


    def step(self, save_document_path = None):
        self._step += 1
        #update the weights
        noises = self._update_weights() # get normal distributions as individual noise for each weight

        # get sample and label
        sample,label = next(self._data_iterator, (None,None)) # move iterator forward, defaults to (None, None) if iterator at end
        # if the iterator reaches the end of the dataset, reinitialise the iterator
        if sample is None:
            self._data_iterator = iter(self._dataloader)
            sample, label = next(self._data_iterator, (None, None))

        with tf.GradientTape(persistent=True) as tape: # TODO: figure out what this does
            # take the posterior distribution into account in the calculation of the gradients
            tape.watch(self._posterior_mean_list)
            tape.watch(self._posterior_std_dev_list)
            predictions = self._base_model(sample, training = True) # get prediction for sample

            # step 4
            likelihood = self._cost_function(
                label, 
                predictions
            ) # calculate cost (i.e. datalikelihood + weighted KL divergence)
        # get the weight, mean and standard deviation gradients
        mean_gradients = tape.gradient(likelihood, self._posterior_mean_list) # get gradient for mean of each weight TODO: make sure this is right
        std_dev_gradients = tape.gradient(likelihood, self._posterior_std_dev_list) # get gradient for std of each weight TODO: make sure this is right

        # save the model loss if the path is specified
        if save_document_path != None:
            with open(save_document_path, "a") as losses_file:
                losses_file.write(str(likelihood.numpy())+"\n")

        new_posterior_mean_list = []
        new_posterior_std_dev_list = []
        trainable_layer_index = 0
        gradient_layer_index = 0
        for layer_idx in range(len(self._base_model.layers)): # for each layer in model
            layer = self._base_model.layers[layer_idx] # get layer
            if len(layer.trainable_variables) != 0: # if layer has trainable variables
                # go through weights and biases and merge their gradients into one vector
                _new_mean_layer_weights = []
                _new_std_dev_layer_weights = []

                for i in range(len(layer.trainable_variables)): # for each trainable value
                    # calculate the new mean of the posterior
                    weight_gradients = tape.gradient(likelihood, layer.trainable_variables[i]) # how is this different to line 147??? I think layer.trainable_variables is a distribution and not just a mean

                    # step 5
                    mean_gradient = mean_gradients[trainable_layer_index][i]+weight_gradients # TODO: check what this does
                    _new_mean_layer_weights.append(
                        self._posterior_mean_list[trainable_layer_index][i]-self._lr*mean_gradient
                    )

                    # step 6
                    # calculate the std_dev gradient
                    posterior_std_dev = self._posterior_std_dev_list[trainable_layer_index][i]
                    noise = noises[gradient_layer_index]
                    std_dev_grad = noise / (1 + tf.math.exp(-posterior_std_dev))
                    std_dev_grad *= weight_gradients
                    std_dev_grad += std_dev_gradients[trainable_layer_index][i]
                    # calculate the new standard deviation of the posterior
                    _new_std_dev_layer_weights.append(
                        posterior_std_dev-self._lr*std_dev_grad
                    )
                    gradient_layer_index += 1
                trainable_layer_index += 1

                # step 7
                new_posterior_mean_list.append(_new_mean_layer_weights)
                new_posterior_std_dev_list.append(_new_std_dev_layer_weights)

        #update the posteriors
        self._posterior_mean_list = new_posterior_mean_list
        self._posterior_std_dev_list = new_posterior_std_dev_list

        if self._step % 10:
            validation_iterator = iter((self._dataset.valid_data.batch(self._dataset.valid_size)))
            for val_samples, val_labels in validation_iterator:
                val_preds = self._base_model(val_samples)
                self.val_losses.append(self._dataset.loss()(val_labels, val_preds))

            self.train_losses.append(likelihood)

        return likelihood





    # perturb obtained posteriors by some noise and then assign to model parameters
    def _update_weights(self):
        """
        generate new weights following the posterior distributions
        """
        noises = []
        for interval_idx in range(len(self._layers_intervals)): # for each layer interval, i.e. for each layer with trainable params
            # reshape the vector and update the base model
            start = self._layers_intervals[interval_idx][0] # layer index
            end = self._layers_intervals[interval_idx][1] # layer index
            for layer_idx in range(start, end + 1): # since start == end, layer_idx does one iteratiokn with value of start (i.e. layer index)
                # go through weights and biases of the layer
                for i in range(len(self._base_model.layers[layer_idx].trainable_variables)): # for each trainable param in layer
                    #sample the new wights as a vector
                    w = self._base_model.layers[layer_idx].trainable_variables[i] # get specific trainable weight

                    # step 1 in optimisation procedure
                    noise = tfp.distributions.Normal(
                        tf.zeros(self._posterior_mean_list[interval_idx][i].shape),
                        tf.ones(self._posterior_std_dev_list[interval_idx][i].shape)
                    ).sample() # gets noise value from distribution # TODO: check how this works, what it initialises to 
                    noises.append(noise)

                    # step 2
                    vector_weights = noise * tf.math.softplus(self._posterior_std_dev_list[interval_idx][i]) #TODO: what does softplus do, but seems the new weight is just spread by std
                    vector_weights += self._posterior_mean_list[interval_idx][i] # shift noise by mean

                    new_weights = tf.reshape(vector_weights, w.shape)
                    self._base_model.layers[layer_idx].trainable_variables[i].assign(new_weights) # set posterior of model to updated posterior distribution
        return noises



    def compile_extra_components(self, **kwargs): # called when compiling optimizer; kind of an init function
        """
            Args:
                prior (GaussianPrior): the prior
                prior2 (GuassianPrior): second prior used to create a mixture prior weighted with hyperparameter pi
        """
        self._base_model = tf.keras.models.model_from_json(self._model_config) # load model from model config
        self._prior = kwargs["prior"] # taken from input args to optimizer.compile
        if "prior2" in kwargs:
            self._prior2 = kwargs["prior2"]
        else:  
            self._prior2 = GaussianPrior(0.0,0.0) # use this in test.py currently
        self._lr = self._hyperparameters.lr 
        self._pi = getattr(self._hyperparameters, 'pi', None) if hasattr(self._hyperparameters, 'pi') else 1 # weighting of prior one when combining prior 1 and 2 below
        self._batch_size = int(self._hyperparameters.batch_size)
        if isinstance(self._prior._mean, int) or isinstance(self._prior._mean, float): # not quite sure in what scenario the prior mean wouldnt be int or float
            sign = self._prior._std_dev/abs(self._prior._std_dev)
            self._prior = GaussianPrior( # summing prior 1 and prior 2 weighted according to pi
                self._prior._mean * self._pi + self._prior2._mean * (1 - self._pi),
                sign * math.sqrt((self._prior._std_dev * self._pi)**2 + (self._prior2._std_dev * (1 - self._pi))**2),
            )

        self._alpha = self._hyperparameters.alpha
        self._dataset_setup() # loads training data and sets self._data_iterator to iterator for training data
        self._priors_list = self._prior.get_model_priors(self._base_model) # creates a prior for each trainable parameter of model
        self._init_BBB_arrays() # inits arrays required for BBB; TODO: match this to part of research paper

    def _init_BBB_arrays(self):
        """
        initialises the posterior list, the correlated layers intervals and the trainable layer indices
        """
        trainable_layer_index = 0 # == layer_idx
        for layer_idx in range(len(self._base_model.layers)):
            layer = self._base_model.layers[layer_idx]
            # iterate through weights and biases of the layer
            mean_layer_posteriors = []
            std_dev_layer_posteriors = []

            if len(layer.trainable_variables) != 0:
                for i in range(len(layer.trainable_variables)): # for each trainable variable of layer
                    mean_layer_posteriors.append(self._priors_list[trainable_layer_index][i].mean()) # get mean of prior
                    std_dev_layer_posteriors.append(self._priors_list[trainable_layer_index][i].stddev()) # get stddev of prior
                self._weight_layers_indices.append(layer_idx) # indicate that a layer has trainable parameters
                self._layers_intervals.append([layer_idx, layer_idx]) # not quite certain what this does
                self._posterior_mean_list.append(mean_layer_posteriors) # init posterior means to prior means
                self._posterior_std_dev_list.append(std_dev_layer_posteriors) # init posterior stds to prior stds
            trainable_layer_index += 1



    def result(self) -> BayesianModel:
        model = BayesianModel(self._model_config)
        for layer_mean_list, layer_std_dev_list, idx in zip(self._posterior_mean_list, self._posterior_std_dev_list, range(len(self._weight_layers_indices))):
            for i in range(len(layer_mean_list)):
                layer_mean_list[i] = tf.reshape(layer_mean_list[i], (-1,))
                layer_std_dev_list[i]= tf.reshape(layer_std_dev_list[i], (-1,))
            mean = tf.concat(layer_mean_list, 0)
            std_dev = tf.concat(layer_std_dev_list,0)
            # tf.debugging.check_numerics(mean, "mean")
            # tf.debugging.check_numerics(std_dev, "standard deviation")
            tf_dist = tfp.distributions.Normal(
                tf.reshape(mean, (-1,)),
                tf.math.softplus(tf.reshape(std_dev, (-1,)))
            )
            tf_dist = TensorflowProbabilityDistribution(
                tf_dist
            )

            start_idx = self._weight_layers_indices[idx]
            end_idx = len(self._base_model.layers) - 1
            if idx + 1 < len(self._weight_layers_indices):
                end_idx = self._weight_layers_indices[idx + 1]
            model.apply_distribution(tf_dist, start_idx, start_idx)
        return model, self.train_losses, self.val_losses

    def update_parameters_step(self):
        pass
