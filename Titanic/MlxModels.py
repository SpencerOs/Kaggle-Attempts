import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
from pathlib import Path

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


class MLX_MLP(nn.Module):
    """ Seemingly a simple MLP """
    def __init__(
        self,
        params = None,
        exploratory_params = None
    ):
        super().__init__()
        
        self.ds = params['data_shop']
        self.eval_fn = params['eval_fn']

        self.batch_size = exploratory_params['batch_size']
        self.epochs = exploratory_params['epochs']
        self.loss_fn = nn.losses.cross_entropy

        self.optimizer = exploratory_params['optimizer'](learning_rate=10**exploratory_params['learning_rate'])

        num_layers = exploratory_params['num_layers']
        input_dim = len(self.ds.train_X[0])
        hidden_dim = exploratory_params['hidden_dim']
        output_dim = len(self.ds.train_y[0])


        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

        mx.eval(self.parameters())

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = nn.relu(l(x))
        return self.layers[-1](x)
    
    def calc_loss(self, X, y):
        return self.loss_fn(self(X), y.reshape(-1, 1), reduction="mean")

    def fit(self):
        loss_and_grad_fn = nn.value_and_grad(self, self.calc_loss)

        for e in range(self.epochs):
            tic = time.perf_counter()
            for X, y in self.ds.batch_iterate(self.batch_size):
                loss, grads = loss_and_grad_fn(X, y)
                self.optimizer.update(self, grads)
                mx.eval(self.parameters(), self.optimizer.state)
            
            eval_score = self.predict()
            toc = time.perf_counter()
            if not e % 10:
                print(
                    f"Epoch {e}: Eval score {eval_score},\n"
                    f"{toc - tic:.3f} (s)"
                )

    def predict(self):
        predictions = self(self.ds.test_X)
        y_hat = (predictions > 0.5).astype(mx.float32)
        mx.eval(self.parameters())
        return self.eval_fn(self.ds.test_y, y_hat.tolist())

    def save_model(self, filename):
        # I'll have to look up just how we commit an MLX model to the drive using the filename string
        flat_params = tree_flatten(self.parameters())
        mx.savez(f"{filename}.npz", **dict(flat_params))