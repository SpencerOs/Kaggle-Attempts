import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import mnist

from abc import ABC, abstractmethod

class MLX_Model(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __calL__(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @property
    @abstractmethod
    def eval(self):
        pass

class MLX_MLP(nn.Module):
    """ Seemingly a simple MLP """
    def __init__(
        self,
        params = None,
        exploratory_params = None
    ):
        super().__init__()
        
        self.data_shop = params['data_shop']
        self.eval_fn = params['eval_fn']
        self.batch_size = exploratory_params['batch_size']
        self.loss_and_grad_fn = exploratory_params['loss_and_grad_fn']
        self.optimizer = exploratory_params['optimizer']

        num_layers = exploratory_params['num_layers']
        input_dim = exploratory_params['input_dim']
        hidden_dim = exploratory_params['hidden_dim']
        output_dim = exploratory_params['output_dim']


        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

        # make it so
        mx.eval(self.parameters())

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = nn.relu(l(x))
        return self.layers[-1](x)

    def fit(self):
        for X, y in self.data_shop.batch_iterate():
            loss, grads = self.loss_and_grad_fn(self.model, X, y)
            self.optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), self.optimizer.state)
        accuracy = self.eval_fn(self.model, )
        result = self(self.data_shop)

    def predict(self):
        return self.model(self.data_shop.submission_set)