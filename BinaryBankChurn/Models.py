import pandas as pd
import tensorflow as tf
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam, RMSprop, SGD
from tensorflow.keras.losses import BinaryCrossentropy

from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def __init__(self):
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

class XGBoostModel(Model):
    def __init__(self, params=None, exploratory_params=None):
        super().__init__()

        params['booster'] = params['booster'] if 'booster' in params else 'gbtree'
        params['device'] = params['device'] if 'device' in params else 'gpu'
        params['verbosity'] = params['verbosity'] if 'verbosity' in params else 1
        
        self.data_shop = params['data_shop']
        self.eval_fn = params['eval_fn']

        self.model = xgb.XGBClassifier(
            booster=params['booster'],
            device=params['device'],
            verbosity=params['verbosity']
        )

        if exploratory_params is not None:
            self.model.set_params(**exploratory_params)

    def fit(self):
        self.model.fit(self.data_shop.train_X, 
                       self.data_shop.train_y)

    def predict(self, X_test=None):
        if X_test is None:
            self.y_pred = self.model.predict(
                self.data_shop.test_X)
        else:
            return self.model.predict(X_test)
        

    def save_model(self, filename):
        self.model.save_model(f'{filename}.json')
    
    @property
    def eval(self):
        eval = None
        if self.y_pred is not None:
            eval = self.eval_fn(self.data_shop.test_y, self.y_pred)
        return eval
    
class ForwardModel(Model):
    def __init__(self, params=None, exploratory_params=None):
        super().__init__()

        self.data_shop = params['data_shop']
        self.eval_fn = params['eval_fn']
        self.expl_params = exploratory_params

        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(self.data_shop.train_X.shape[1],)),
            Dropout(exploratory_params['dropout_rate'])
        ])

        for _ in range(exploratory_params['num_layers']):
            self.model.add(Dense(exploratory_params['neurons_per_layer'], activation='relu'))
            self.model.add(Dropout(exploratory_params['dropout_rate']))

        self.model.add(Dense(1, activation='sigmoid'))

        if exploratory_params['optimizer'] == 'adam':
            optimizer = Adam(learning_rate=exploratory_params['learning_rate'])
        elif exploratory_params['optimizer'] == 'rmsprop':
            optimizer = RMSprop(learning_rate=exploratory_params['learning_rate'])
        else:
            optimizer = SGD(learning_rate=exploratory_params['learning_rate'])

        self.model.compile(optimizer=optimizer, loss=BinaryCrossentropy(), metrics=[tf.keras.metrics.AUC()])

    def fit(self):
        self.model.fit(self.data_shop.train_X,
                       self.data_shop.train_y,
                       validation_data = (self.data_shop.test_X, 
                                          self.data_shop.test_y),
                        epochs=self.expl_params['epochs'],
                        batch_size=self.expl_params['batch_size'],
                        verbose=0)
        
    def predict(self, X_test=None):
        if X_test is None:
            self.y_pred = self.model.predict(
                self.data_shop.test_X)
        else:
            result = self.model.predict(X_test).flatten()
            result = pd.DataFrame(result).iloc[:,0]
            return result
        
    def save_model(self, filename):
        self.model.save(f'{filename}.h5')

    @property
    def eval(self):
        eval = None
        if hasattr(self, 'y_pred'):
            eval=self.eval_fn(self.data_shop.test_y, self.y_pred)
        return eval