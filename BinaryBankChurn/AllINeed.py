import pandas as pd
import pickle
import xgboost as xgb
from lifelines import CoxPHFitter

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
        self.model.fit(self.data_shop.training_split_X, 
                       self.data_shop.training_split_y)

    def predict(self, X_test=None):
        if X_test is None:
            self.y_pred = self.model.predict(
                self.data_shop.testing_split_X)
        else:
            return self.model.predict(X_test)

    
    def save_model(self, filename):
        self.model.save_model(f'{filename}.json')
    
    @property
    def eval(self):
        eval = None
        if self.y_pred is not None:
            eval = self.eval_fn(self.data_shop.testing_split_y, self.y_pred)
        return eval

class CoxPHModel(Model):
    def __init__(self, params=None, exploratory_params=None):
        super().__init__()

        self.data_shop = params['data_shop']
        self.eval_fn = params['eval_fn']
        self.duration_col = params['duration_col']
        self.event_col = params['event_col']

        if exploratory_params is not None:
            self.model = CoxPHFitter(exploratory_params)
        else:
            self.model = CoxPHFitter()

    def fit(self):
        self.model.fit(self.data_shop.train_frame, 
                       duration_col=self.duration_col, 
                       event_col=self.event_col)
    
    def predict(self, X_test=None):
        if X_test is None:
            self.y_pred = self.model.predict(
                self.data_shop.testing_split_X)
        else:
            return self.model.predict_survival_function(X_test)
        

    def save_model(self, filename):
        with open(f'{filename}.pkl', 'wb') as file:
            pickle.dump(self.model, file)
    
    @property
    def eval(self):
        eval = None
        if self.y_pred is not None:
            eval = self.eval_fn(self.data_shop.testing_split_y, self.y_pred)
        return eval