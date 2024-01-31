from hyperopt import fmin, tpe, Trials
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from typing import Any

class MlxNavigator:
    def __init__(self, space, model_class, data_shop, eval_metric):
        self.space = space
        self.model_class = model_class
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
        model.predict()
        # Take your new shape
        eval_score = 1 - model.eval
        return eval_score
        
    def explore_and_learn(self, max_evals:int):
        best = fmin(
            fn=self.objective, 
            space=self.space, 
            algo=tpe.suggest, 
            max_evals=max_evals, 
            trials=Trials())

        best['loss_fn'] = self.space['loss_fn'][best['loss_fn']]
        best['optimizer'] = self.space['optimizer'][best['optimizer']]
        # intersecting_best = {k:best[k] for k in best if k in self.space}
        print(f"model_params going into the last fitting: \n{self.model_params}\n{self.model_params['data_shop'].train_X}")
        model_reshaped = self.model_class(params=self.model_params, exploratory_params=best)
        model_reshaped.fit()
        return model_reshaped