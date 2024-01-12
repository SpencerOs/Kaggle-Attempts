from hyperopt import fmin, tpe, Trials
from sklearn.metrics import f1_score, recall_score

class TheNavigator:
    def __init__(self, space, model_class, data_shop, eval_metric):
        self.space = space
        self.model_class = model_class
        self.data_shop = data_shop
        if eval_metric == 'f1_score':
            self.eval_metric = f1_score
        elif eval_metric == 'recall_score':
            self.eval_metric = recall_score

    def objective(self, expl_params):
        # There is a knife for you. It is shaped like [X, y]
        params = {
            'data_shop': self.data_shop,
            'eval_fn': self.eval_metric
        }
        model = self.model_class(params=params, exploratory_params=expl_params)

        # Take up the knife. 
        model.fit()
        # ŷ the data. 
        model.predict()
        # Take your new shape.
        return -model.eval # Negative eval score to shoot for minimization
    

    def explore_and_learn(self, max_evals:int):
        best = fmin(fn=self.objective, space=self.space, algo=tpe.suggest, max_evals=max_evals, trials=Trials())
        intersecting_best = {k: best[k] for k in best if k in self.space}
        model_reshaped = self.model_class(params={'data_shop':self.data_shop}, exploratory_params=intersecting_best)
        model_reshaped.fit()
        return model_reshaped