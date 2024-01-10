import argparse
import pandas as pd

from hyperopt import hp

from AllINeed import XGBoostModel, CoxPHModel
from DataShop import DataShop
from TheNavigator import TheNavigator

cph_name = 'cph'
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
    elif args.model == cph_name:
        return {
            'l1_ratio': hp.uniform('l1_ratio', 0.0, 1.0)
        }
    
def get_model_class(args):
    if args.model == xgboost_name:
        return XGBoostModel
    elif args.model == cph_name:
        return CoxPHModel
    

def main(args):
    ds = DataShop(train_file="train.csv", test_file="test.csv")

    identifiers = ['CustomerId', 'Surname']
    numerical = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    booleans = ['HasCrCard', 'IsActiveMember']
    one_hots = ['Geography', 'Gender']
    target = 'Exited'
    test_id = 'id'
    ds.categorize_columns(
        identifiers=identifiers,
        numerical=numerical,
        booleans=booleans,
        one_hots=one_hots,
        target=target,
        test_id=test_id
    )

    X_train, y_train = ds.training_data

    space = get_model_hp_space(args)
    model_class = get_model_class(args)
    navigator = TheNavigator(
        space, 
        model_class, 
        X_train, 
        y_train, 
        'recall_score'
    )
    
    model = navigator.explore_and_learn(max_evals=10000)

    model.save_model(f'bank_churn_{args.model}')

    predictions = model.predict(ds.testing_data)

    submission = pd.DataFrame({
        'id': ds.testing_ids,
        'Exited': predictions
    })
    submission.to_csv(f'{args.model}_prediction.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a model for the Binary Classifier Bank Churn dataset.")
    # Maybe we add something fun here down the road
    parser.add_argument(
        "--model", 
        type=str,
        default=xgboost_name,
        choices=[xgboost_name, cph_name], 
        required=True, 
        help='Which model you\'d like to train on')
    args = parser.parse_args()

    main(args)