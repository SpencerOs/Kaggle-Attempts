import xgboost as xgb

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
        self.model.save_model(filename)