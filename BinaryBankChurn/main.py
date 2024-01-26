import argparse
import pandas as pd

from hyperopt import hp

from AllINeed import XGBoostModel, ForwardModel
from DataShop import DataShop
from TheNavigator import TheNavigator

forward_name = 'forward'
xgboost_name = 'xgboost'

def get_model_hp_space(args):
    if args.model == xgboost_name:
        return {
            'max_depth': hp.choice('max_depth', range(3, 11)),              # Integers between 3 and 10
            'min_child_weight': hp.choice('min_child_weight', range(1, 6)), # Uniformly distributed integers between 1 and 6
            'gamma': hp.uniform('gamma', 0.0, 5.0),                         # Continuous values between 0 and 5
            'subsample': hp.uniform('subsample', 0.5, 1.0),                 # Continuous values between 1/2 and 1
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0)    # Continuous values between 1/2 and 1
        }
    elif args.model == forward_name:
        return {
            'num_layers': hp.choice('num_layers', range(1, 5)),
            'neurons_per_layer': hp.choice('neurons_per_layer', [32, 64, 128, 256]),
            'learning_rate': hp.loguniform('learning_rate', -5, 0),
            'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.75),
            'optimizer': hp.choice('optimizer', ['adam', 'rmsprop', 'sgd']),
            'batch_size': hp.choice('batch_size', [32, 64, 128]),
            'epochs': hp.choice('epochs', [10, 20, 50, 1008])
        }
    
def get_model_class(args):
    if args.model == xgboost_name:
        return XGBoostModel
    elif args.model == forward_name:
        return ForwardModel
    

def main(args):
    ds = DataShop(train_file="train.csv", test_file="test.csv")

    identifiers = ['CustomerId', 'Surname']
    numerical = ['CreditScore', 'Age', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    if args.model == xgboost_name:
        numerical.append('Tenure')
    booleans = ['HasCrCard', 'IsActiveMember']
    one_hots = ['Geography', 'Gender']
    non_proc = []
    target = 'Exited'
    test_id = 'id'
    ds.categorize_columns(
        identifiers=identifiers,
        numerical=numerical,
        booleans=booleans,
        one_hots=one_hots,
        non_proc=non_proc,
        target=target,
        test_id=test_id
    )

    space = get_model_hp_space(args)
    model_class = get_model_class(args)
    navigator = TheNavigator(
        space, 
        model_class, 
        ds, 
        args.eval_metric
    )
    
    model = navigator.explore_and_learn(max_evals=10000)

    model.save_model(f'bank_churn_{args.model}')

    predicted_outcome = model.predict(ds.submission_set)

    submission = pd.DataFrame({
        'id': ds.testing_ids,
        'Exited': predicted_outcome
    })
    submission.to_csv(f'{args.model}_prediction.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a model for the Binary Classifier Bank Churn dataset.")
    # Maybe we add something fun here down the road
    parser.add_argument(
        "--model", 
        type=str,
        default=xgboost_name,
        choices=[xgboost_name, forward_name], 
        required=True, 
        help='Which model you\'d like to train on')
    parser.add_argument(
        "--eval_metric",
        type=str,
        default='f1_score',
        choices=['f1_score', 'recall_score', 'roc_auc'],
        help="Which evaluation_metric would you like to use")
    args = parser.parse_args()

    main(args)