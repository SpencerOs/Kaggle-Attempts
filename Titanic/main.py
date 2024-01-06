import argparse
import pandas as pd
import tensorflow as tf
import xgboost as xg

from hyperopt import hp, tpe, Trials, fmin
from pandas import DataFrame
from sklearn.compose import make_column_transformer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline

def process_data(dataframe:DataFrame):
    # Drop the columns that are [hopefully] less relevant to survival (e.g. name)
    dataframe = dataframe.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    # Fill in missing values
    dataframe['Age'].fillna(dataframe['Age'].mean(), inplace=True)
    dataframe['Embarked'].fillna('S', inplace=True)

    return dataframe

def fetch_training_data():
    # Load the training dataset
    train_df = pd.read_csv('train.csv')
    train_df = process_data(train_df)
    
    # Split the datasets
    X_train = train_df.drop('Survived', axis=1)
    y_train = train_df['Survived']

    return X_train, y_train

def fetch_testing_dataframe():
    # Load the testing dataset
    test_df = pd.read_csv('test.csv')

    test_df = process_data(test_df)

    return test_df

def create_data_transformer(X_train):
    # Preprocessing pipelines for both numerical and categorical data
    numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X_train.select_dtypes(include=['object', 'bool']).columns

    numerical_pipeline = make_pipeline(StandardScaler())
    categorical_pipeline = make_pipeline(OneHotEncoder(handle_unknown='ignore'))

    preprocessor = make_column_transformer(
        (numerical_pipeline, numerical_cols),
        (categorical_pipeline, categorical_cols)
    )
    return preprocessor

# Create the MLP model
def create_mlp_model(input_shape, learning_rate=0.001):
    model = tf.keras.Sequential()

    # Add layers to the model
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# XGBoost is all you need
def create_xgb_model():
    model = xg.XGBClassifier(
        objective='binary:logistic', 
        eval_metric='logloss', 
        use_label_encoder=False
    )
    return model

def fit_dataset_preprocessor(X_train, preprocessor):
    X_train_processed = preprocessor.fit_transform(X_train)
    return X_train_processed, preprocessor

def preprocess_dataset(dataframe, preprocessor):
    dataframe_processed = preprocessor.transform(dataframe)

    return dataframe_processed

def objective(params, X_train, y_train):
    # Split the training data for validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    # Create and train the model
    model = create_mlp_model(input_shape=(X_train.shape[1],), learning_rate=params['learning_rate'])
    model.fit(X_train, y_train, epochs=int(params['epochs']))

    # Predict on validation set and calculate F1-score
    val_predictions = model.predict(X_val)
    val_predictions = (val_predictions > params['threshold']).astype(int)
    f1 = f1_score(y_val, val_predictions)

    return -f1  # Negative F1-score for minimization

def main(run_mlp:bool):
    X_train, y_train = fetch_training_data()
    preprocessor = create_data_transformer(X_train)
    test_df = fetch_testing_dataframe()

    X_train_processed, preprocessor = fit_dataset_preprocessor(X_train, preprocessor)
    X_submission_processed = preprocess_dataset(test_df, preprocessor)

    if run_mlp:
        space = {
            'learning_rate': hp.uniform('learning_rate', 0.00001, 0.2),
            'epochs': hp.quniform('epochs', 10, 100, 10),
            'threshold': hp.uniform('threshold', 0.2, 0.7)
        }

        def objective_wrapper(params):
            return objective(params, X_train_processed, y_train)

        best = fmin(fn=objective_wrapper, space=space, algo=tpe.suggest, max_evals=100, trials=Trials())
        
        # Train the final model with the best hyperparameters
        model = create_mlp_model(input_shape=(X_train_processed.shape[1],), learning_rate=best['learning_rate'])
        model.fit(X_train_processed, y_train, epochs=int(best['epochs']))

        model.save('titanic_mlp_model.keras')
    else:
        model = create_xgb_model()
        model.fit(X_train_processed, y_train)
        model.save_model('xgboost_model.json')

    # Generate predictions on test data
    predictions = model.predict(X_submission_processed)

    # Prepare and save the submission file
    if run_mlp:
        best_threshold = best['threshold']
        predictions = (predictions > best_threshold).astype(int)

        submission = pd.DataFrame({
            'PassengerId': test_df['PassengerId'],
            'Survived': predictions.ravel()
        })
        submission.to_csv('mlp_submission.csv', index=False)
    else:
        # Prepare submission file
        submission = pd.DataFrame({
            'PassengerId': test_df['PassengerId'],
            'Survived': predictions
        })
        submission.to_csv('xgboost_submission.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a model on the Kaggle Titanic dataset using TensorFlow!")
    parser.add_argument("--model", choices=['mlp', 'xgboost'], required=True, help='The architecture of the model you\'d like to train')
    args = parser.parse_args()

    main(args.model == 'mlp')