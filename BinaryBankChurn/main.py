import argparse
import pandas as pd

from hyperopt import hp

from AllINeed import XGBoostModel
from DataShop import DataShop
from TheNavigator import TheNavigator

def main():
    ds = DataShop(train_file="train.csv", test_file="test.csv")

    identifiers = ['CustomerId', 'Surname', 'id']
    numerical = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    one_hots = ['Geography', 'Gender']
    target = 'Exited'
    test_id = 'id'
    ds.categorize_columns(
        identifiers=identifiers,
        numerical=numerical,
        one_hots=one_hots,
        target=target,
        test_id=test_id
    )

    X_train, y_train = ds.training_data

    space = {
        'max_depth': hp.choice('max_depth', range(3, 11)),              # Integers between 3 and 10
        'min_child_weight': hp.choice('min_child_weight', range(1, 6)), # Uniformly distributed integers between 1 and 6
        'gamma': hp.uniform('gamma', 0.0, 5.0),                         # Continuous values between 0 and 5
        'subsample': hp.uniform('subsample', 0.5, 1.0),                 # Continuous values between 1/2 and 1
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0)    # Continuous values between 1/2 and 1
    }

    navigator = TheNavigator(
        space, 
        XGBoostModel, 
        X_train, 
        y_train, 
        'recall_score'
    )
    
    model = navigator.explore_and_learn(max_evals=10000)

    model.save_model('bank_churn_xgboost.json')

    predictions = model.predict(ds.testing_data)

    submission = pd.DataFrame({
        'id': ds.testing_ids,
        'Exited': predictions
    })
    submission.to_csv('xgboost_prediction.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a model for the Binary Classifier Bank Churn dataset.")
    # Maybe we add something fun here down the road
    argsg = parser.parse_args()

    main()