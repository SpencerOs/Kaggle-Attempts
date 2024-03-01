import time
from functools import partial

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

class Test_MLP(nn.Module):
    def __init__(
        self,
        params = None,
        exploratory_params = None
    ):
        super().__init__()

        self.ds = params['data_shop']

        self.batch_size = 2
        self.epochs = exploratory_params['epochs']
        self.loss_fn = nn.losses.binary_cross_entropy

        self.optimizer = optim.Adam(learning_rate=10**-4)

        input_dim = len(self.ds.train_X[0])
        output_dim = len(self.ds.train_y[0])
        layer_sizes = [input_dim] + [5] * 3 + [output_dim]
        self.layers = [
            nn.Linear(idim, odim) for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

        mx.eval(self.parameters())

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = nn.relu(l(x))
        return self.layers[-1](x)
    
    def eval_fn(self):
        # Assuming predictions and labels are mlx arrays with your model's output and the true labels, respectively
        # Ensure predictions are in the correct form (e.g., class labels for classification tasks)
        pred = self(self.ds.test_X)
        labels = self.ds.test_y
        labels_reshaped = labels.reshape((133,1))

        # Convert predictions to binary labels (0 or 1) based on a threshold (e.g., 0.5 for binary classification)
        # This step may vary based on your specific model output and task
        threshold = 0.5
        binary_predictions = pred > threshold

        # Calculate accuracy
        # First, compare predictions to labels to get a boolean array of correct predictions
        print(f"Shape differences-------------\nbinary predictions {binary_predictions.shape}\nlabels {labels_reshaped.shape}")
        correct_predictions = binary_predictions == labels_reshaped

        # Then, calculate the mean of correct predictions. Since True evaluates to 1 and False to 0,
        # taking the mean gives the proportion of correct predictions, which is the accuracy
        accuracy = mx.mean(correct_predictions)

        # Convert accuracy to a more readable format if necessary (e.g., a Python scalar)
        accuracy_value = accuracy.item()  # This step might vary based on MLX's specific API for handling scalar values

        print(f"Model accuracy: {accuracy_value * 100:.2f}%")
        return accuracy_value

    def fit(self):
        def train_step(model, X, y):
            pred = model(X)

            loss = mx.mean(nn.losses.binary_cross_entropy(pred, y))
            acc = mx.mean(mx.argmax(pred, axis=1) == y)
            return loss, acc
        
        losses = []
        accs = []
        samples_per_sec = []

        state = [self.state, self.optimizer.state]
        
        @partial(mx.compile, inputs=state, outputs=state)
        def step(input, target):
            train_step_fn = nn.value_and_grad(self, train_step)
            (loss, acc), grads = train_step_fn(self, input, target)
            self.optimizer.update(self, grads)
            return loss, acc
        
        for e in range(self.epochs):
            for batch_counter, (X, y) in enumerate(self.ds.batch_iterate(self.batch_size)):
                tic = time.perf_counter()
                loss, acc = step(X, y)
                mx.eval(state)
                toc = time.perf_counter()
                loss = loss.item()
                acc = acc.item()
                losses.append(loss)
                accs.append(acc)
                throughput = X.shape[0]/(toc-tic)
                samples_per_sec.append(throughput)
                if batch_counter % 10 == 0:
                    print(
                        " | ".join(
                            (
                                f"Epoch {e:02d} [{batch_counter:03d}]",
                                f"Train loss: {loss:.3f}",
                                f"Train acc: {acc:.3f}",
                                f"Throughput: {throughput:.2f} samples/second"
                            )
                        )
                    )

            mean_tr_loss = mx.mean(mx.array(losses))
            mean_tr_acc = mx.mean(mx.array(accs))
            samples_per_sec = mx.mean(mx.array(samples_per_sec))
            print(
                "\n".join(
                    (
                        f"Epoch {e:02d}:-------",
                        f"mean tr loss: {mean_tr_loss}",
                        f"mean tr acc: {mean_tr_acc}"
                    )
                )
            )

            eval_score = self.predict()
            print(f"Epoch: {e} | eval_score: {eval_score.item():.3f}")
        
        # loss_and_grad_fn = nn.value_and_grad(self, self.calc_loss)

        # for e in range(self.epochs):
        #     for X, y in self.ds.batch_iterate(self.batch_size):
        #         loss, grads = loss_and_grad_fn(X, y)
        #         self.optimizer.update(self, grads)
        #         mx.eval(self.parameters(), self.optimizer.state)
        #     eval_score = self.predict()
        
    def predict(self):
        acc = self.eval_fn()
        acc_value = acc.item()
        return acc_value
    
    def save_model(self, filename):
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
        # predictions = self(X)
        print("About to go through predictions:")
        print(f"Logits: {mx.flatten(self(X))}\nTargets: {y}\n\n")
        # y_hat = (predictions > 0.5).astype(mx.float32)
        return mx.mean(self.loss_fn(mx.flatten(self(X)), y))

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
        y_hat = [item[0] for item in y_hat.tolist()]
        return self.eval_fn(self.ds.test_y, y_hat)

    def save_model(self, filename):
        # I'll have to look up just how we commit an MLX model to the drive using the filename string
        flat_params = tree_flatten(self.parameters())
        mx.savez(f"{filename}.npz", **dict(flat_params))