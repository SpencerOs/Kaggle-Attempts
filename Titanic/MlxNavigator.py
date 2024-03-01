from hyperopt import fmin, tpe, Trials
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from typing import Any

class MlxNavigator:
    def __init__(self, space, model_class, model_hp_choices, data_shop, eval_metric):
        self.space = space
        self.model_class = model_class
        self.model_hp_choices = model_hp_choices
        self.data_shop = data_shop
        if eval_metric == 'accuracy': 
            self.eval_metric = accuracy_score
        elif eval_metric == 'f1_score':
            self.eval_metric = f1_score
        elif eval_metric == 'recall_score':
            self.eval_metric = recall_score
        elif eval_metric == 'roc_auc':
            self.eval_metric = roc_auc_score

        self.model_params = {
            'data_shop': self.data_shop,
            'eval_fn': self.eval_metric
        }

    def objective(self, expl_params):
        # There is a knife for you. It is shaped like [X, y]
        model = self.model_class(params=self.model_params, exploratory_params=expl_params)
        # Take up the knife
        model.fit()
        # ŷ the data.
        model_eval = model.predict()
        # Take your new shape
        eval_score = 1 - model_eval
        return eval_score
        
    def explore_and_learn(self, max_evals:int):
        best = fmin(
            fn=self.objective, 
            space=self.space, 
            algo=tpe.suggest, 
            max_evals=max_evals, 
            trials=Trials())

        best_hyperspot = {}
        for key, value in best.items():
            if key in self.model_hp_choices:
                best_hyperspot[key] = self.model_hp_choices[key][value]
            else:
                best_hyperspot[key] = value
        model_reshaped = self.model_class(params=self.model_params, exploratory_params=best_hyperspot)
        model_reshaped.fit()
        return model_reshaped