from hyperopt import fmin, tpe, Trials
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split

class TheNavigator:
    def __init__(self, space, model_class, X_train, y_train, eval_metric):
        self.space = space
        self.model_class = model_class
        self.X_train = X_train
        self.y_train = y_train
        self.eval_metric = eval_metric

    def objective(self, params):
        # Therre is a knife for you. It is shaped like [X_train, y_train]
        X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size=0.111)

        # Take up the knife. Reflect the data. Take your new shape.
        model = self.model_class(exploratory_params=params)
        model.fit(X_train, y_train)

        val_predictions = model.predict(X_val)
        if self.eval_metric == 'f1_score':
            eval = f1_score(y_val, val_predictions)
        elif self.eval_metric == 'recall_score':
            eval = recall_score(y_val, val_predictions)
        
        return -eval # Negative eval score to shoot for minimization
    

    def explore_and_learn(self, max_evals:int):
        best = fmin(fn=self.objective, space=self.space, algo=tpe.suggest, max_evals=max_evals, trials=Trials())
        intersecting_best = {k: best[k] for k in best if k in self.space}
        model_reshaped = self.model_class(exploratory_params=intersecting_best)
        model_reshaped.fit(self.X_train, self.y_train)
        return model_reshaped