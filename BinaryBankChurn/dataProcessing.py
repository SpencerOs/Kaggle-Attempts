import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

def process_data(dataframe:DataFrame):
    # We're going to assume that there will be some attributes to rows which will not end up affecting the churning
    dataframe = dataframe.drop(['CustomerId', 'Surname', 'id'], axis=1)
    
    # Geography has three unique values, and Gender has two, so we'll one-hot encode them
    dataframe = pd.get_dummies(dataframe, columns=['Geography', 'Gender'])

    # Get a hold of the columns which rep numerical values and get them nice & normalized
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    scaler = StandardScaler()
    dataframe[numerical_cols] = scaler.fit_transform(dataframe[numerical_cols])

    return dataframe, processor

def fetch_training_data():
    # Load in the testing dataset
    train_df = pd.read_csv('train.csv')
    train_df = process_data(train_df)

    X_train = train_df.drop('Exited', axis=1)
    y_train = train_df['Exited']
    return X_train, y_train