import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

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
        self.epochs = exploratory_params['epochs']
        self.loss_fn = exploratory_params['loss_fn']

        self.optimizer = exploratory_params['optimizer'](learning_rate=10**exploratory_params['learning_rate'])

        num_layers = exploratory_params['num_layers']
        # input_dim = len(params['data_shop'].train_X[0])
        print(f'ds: {dir(params["data_shop"])}')
        print(f'train_X: {params["data_shop"].train_X}')
        hidden_dim = exploratory_params['hidden_dim']
        # output_dim = len(params['data_shop'].train_y[0])
        print(f'ds.train_y: {len(params["data_shop"].train_y[0])}')


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
        loss_and_grad_fn = nn.value_and_grad(self, self.loss_fn)

        for e in range(self.epochs):
            tic = time.perf_counter()
            for X, y in self.data_shop.batch_iterate():
                loss, grads = loss_and_grad_fn(self.model, X, y)
                self.optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), self.optimizer.state)
            
            eval_score = self.eval_fn(self.data_shop.test_y, self(self.data_shop.test_X))
            toc = time.perf_counter()
            print(
                f"Epoch {e}: Eval score {eval_score},\n"
                f"{toc - tic:.3f} (s)"
            )


    def predict(self):
        self.y_pred = self.model(self.data_shop.test_X)
        mx.eval(self.model.parameters())

    def save_model(self, filename):
        # I'll have to look up just how we commit an MLX model to the drive using the filename string
        pass

    @property
    def eval(self):
        eval = None
        if hasattr(self, 'y_pred'):
            eval = self.eval_fn(self.data_shop.test_y, self.y_pred)
        return eval