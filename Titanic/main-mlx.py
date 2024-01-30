import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import argparse
import pandas as pd
from hyperopt import hp

from Titanic.DataShopDx import DataShop
from MlxModels import MLX_MLP
from MlxNavigator import MlxNavigator

problem_name = 'titanic'

mlp_name = 'mlp'

def get_model_hp_space(args):
    if args.model == mlp_name:
        return {
            'num_layers': hp.choice('num_layers', range(3, 11)),
            'learning_rate': hp.loguniform('learning _rate', -5, 0),
            'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.75),
            'optimizer': hp.choice('optimizer', [optim.Adam, optim.SGD, optim.RMSprop]),
            'batch_size': hp.choice('batch_size', [32, 64, 128]),
            'epochs': hp.choice('epochs', [10, 20, 30, 50])
        }
    
def get_model_class(args):
    if args.model == mlp_name:
        return MLX_MLP
    
def main(args):
    # Identifiers
    #   'PassengerId'
    # # Labels
    #   'Survived'
    # # Features
    #   'Pclass'
    #   'Sex' 
    #   'Age'
    #   'SibSp'
    #   'Parch'
    #   'Fare'
    #   'Cabin'
    #   'Embarked'
    # # To Discard
    #   'Name'
    #   'Ticket'
    identifiers = ['PassengerId']
    target = 'Survived'

    meta = {
        'identifiers': identifiers,
        'target_name': target
    }
    ds = DataShop(train_file="train.csv", test_file="test.csv", meta=meta)

    space = get_model_hp_space(args)
    model_class = get_model_class(args)
    navigator = MlxNavigator(
        space=space,
        model_class=model_class,
        data_shop=ds,
        eval_metric=args.eval_metric
    )

    model = navigator.explore_and_learn(max_evals=10)
    model.save_model(f'{problem_name}_{args.model}_mlx')

    predicted_outcome = model(ds.submission_set)
    submission = pd.DataFrame({
        'PassengerId': ds.testing_ids,
        'Survived': predicted_outcome
    })
    submission.to_csv(f'{problem_name}_{args.model}_mlx_prediction.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train an MLX model for the Titanic dataset.")
    parser.add_argument(
        "--model",
        type=str,
        defaut=mlp_name,
        choices=[mlp_name],
        help='Which model you\'d like to train on')
    parser.add_argument(
        '--eval_metric',
        type=str,
        default='accuracy',
        choices=['accuracy', 'f1_score', 'recall_score', 'roc_auc'],
        help="Which evaluation_metric would you like to use?")
    args = parser.parse_args()

    main(args)