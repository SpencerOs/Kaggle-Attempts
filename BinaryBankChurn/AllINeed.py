import pandas as pd
import pickle
import xgboost as xgb
from lifelines import CoxPHFitter

class XGBoostModel:
    def __init__(self, params=None, exploratory_params=None):
        if params is None:
            params = {
                'booster': 'gbtree',
                'device': 'gpu',
                'verbosity': 1
            }

        self.model = xgb.XGBClassifier(
            **params
        )

        if exploratory_params is not None:
            self.model.set_params(**exploratory_params)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def save_model(self, filename):
        self.model.save_model(f'{filename}.json')

class CoxPHModel:
    def __init__(self, params=None, exploratory_params=None):
        if exploratory_params is not None:
            self.model = CoxPHFitter(exploratory_params)
        else:
            self.model = CoxPHFitter()

    def fit(self, X_train, y_train):
        training_data = pd.concat([X_train, y_train], axis=1)
        self.model.fit(training_data, duration_col='Tenure', event_col='Exited')
    
    def predict(self, X_test):
        return self.model.predict_survival_function(X_test)

    def save_model(self, filename):
        with open(f'{filename}.pkl', 'wb') as file:
            pickle.dump(self.model, file)    