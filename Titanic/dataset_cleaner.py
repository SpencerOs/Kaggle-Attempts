import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Example usage:
file_path = 'test.csv'
file_output_path = 'test_cleaned.csv'
columns_to_remove = ['Name', 'Ticket', 'Cabin']
columns_to_encode = ['Sex', 'Embarked']
columns_to_impute = ['Age', 'Fare']
columns_to_normalize = ['Age', 'Fare']

def remove_columns_from_csv(
        df,
        columns_to_remove):
    df = df.drop(columns=columns_to_remove, errors='ignore')
    return df
    

def one_hot_encode_columns(df, columns_to_encode):
    # ‘pd.get_dummies’ automatically handles the encoding and removes the original columns
    df = pd.get_dummies(df, columns=columns_to_encode)
    return df

def normalize_columns(df, columns_to_normalize):
    scaler = MinMaxScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df

def impute_columns(df, columns_to_impute):
    for col in columns_to_impute:
        df[col].fillna(df[col].median(), inplace=True)
    return df

def integerize_bool_columns(df):
    for column in df.columns:
        if df[column].dtype == 'bool':
            df[column] = df[column].astype(int)
        elif df[column].dtype == object and df[column].isin(['True', 'False']).any():
            df[column] = df[column].replace({'True': 1, 'False': 0})
    return df

df = pd.read_csv(file_path)

df = remove_columns_from_csv(df, columns_to_remove)
df = one_hot_encode_columns(df, columns_to_encode)
df = normalize_columns(df, columns_to_normalize)
df = impute_columns(df, columns_to_impute)
df = integerize_bool_columns(df)

df.to_csv(file_output_path, index=False)