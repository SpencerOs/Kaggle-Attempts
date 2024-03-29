import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import argparse
import pandas as pd
from hyperopt import hp

from DataShopDx import DataShopDx
from MlxModels import MLX_MLP, Test_MLP
from MlxNavigator import MlxNavigator

problem_name = 'titanic'

mlp_name = 'mlp'
test_name = 'test'

def get_model_hp_choices(name):
    if name == mlp_name:
        return {
            'batch_size': [32, 64, 128],
            'epochs': [10, 20, 30, 50],
            'hidden_dim': range(6, 36),
            # 'loss_fn': [nn.losses.cross_entropy, nn.losses.mse_loss],
            'num_layers': range(3, 11),
            'optimizer': [optim.Adam, optim.SGD, optim.RMSprop]
        }
    elif name == test_name:
        return {
            'epochs': [3, 5, 7]
        }

def get_model_hp_space(args):
    if args.model == mlp_name:
        space = {
            'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.75),
            'learning_rate': hp.loguniform('learning_rate', -5, -2),
        }
    else:
        space = {}
    for key, value in get_model_hp_choices(args.model).items():
        space[key] = hp.choice(key, value)
    return space
    
def get_model_class(args):
    if args.model == mlp_name:
        return MLX_MLP
    if args.model == test_name:
        return Test_MLP
    
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

    meta = {
        'identifiers': 'PassengerId',
        'target_name': 'Survived'
    }
    ds = DataShopDx(train_file="train_cleaned.csv", test_file="test_cleaned.csv", meta=meta)

    space = get_model_hp_space(args)
    model_class = get_model_class(args)
    navigator = MlxNavigator(
        space=space,
        model_class=model_class,
        model_hp_choices=get_model_hp_choices(args.model),
        data_shop=ds,
        eval_metric=args.eval_metric
    )

    model = navigator.explore_and_learn(max_evals=1)
    model.save_model(f'{problem_name}_{args.model}_mlx')

    predicted_outcome = model(ds.submission_set)
    mx.eval(model.parameters())
    flattened_prediction = [item[0] for item in predicted_outcome.tolist()]
    submission = pd.DataFrame({
        'PassengerId': ds.testing_ids,
        'Survived': flattened_prediction
    })
    submission.to_csv(f'{problem_name}_{args.model}_mlx_prediction.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train an MLX model for the Titanic dataset.")
    parser.add_argument(
        "--model",
        type=str,
        default=test_name,
        choices=[mlp_name, test_name],
        help='Which model you\'d like to train on')
    parser.add_argument(
        '--eval_metric',
        type=str,
        default='accuracy',
        choices=['accuracy', 'f1_score', 'recall_score', 'roc_auc'],
        help="Which evaluation_metric would you like to use?")
    args = parser.parse_args()

    main(args)