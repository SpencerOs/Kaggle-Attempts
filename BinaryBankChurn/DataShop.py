import pandas as pd
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataShop:
    # Easy as 1...
    def __init__(self, train_file:str, test_file:str):
        try:
            self.train_df = pd.read_csv(train_file)
            self.test_df = pd.read_csv(test_file)
        except FileNotFoundError:
            print("One or both files were not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    # 2...
    def categorize_columns(self, 
                            identifiers:list[str]=None, 
                            numerical:list[str]=None, 
                            booleans:list[str]=None,
                            one_hots:list[str]=None,
                            non_proc:list[str]=None,
                            target:str=None,
                            test_id:str=None):
        self.identifier_col_names = identifiers
        self.numerical_col_names = numerical
        self.boolean_col_names = booleans
        self.one_hot_col_names = one_hots
        self.non_proc_col_names = non_proc
        self.target_col_name = target
        self.test_id_col_name = test_id

        self.process_train_and_test()

    def process_train_and_test(self):
        X_train, y_train = self.clean_inputs(self.train_df, training=True)
        self.test_ids = self.test_df[self.test_id_col_name]
        test_df = self.clean_inputs(self.test_df)

        X_train_transform = X_train.drop(columns=self.non_proc_col_names)
        X_pred_transform = test_df.drop(columns=self.non_proc_col_names)

        self.create_data_transformer()
        X_train_transform = self.preprocessor.fit_transform(X_train_transform)
        X_pred_transform = self.preprocessor.transform(X_pred_transform)

        transformed_col_names = self.preprocessor.get_feature_names_out()
        X_train_transform = DataFrame(X_train_transform, columns=transformed_col_names)
        X_pred_transform = DataFrame(X_pred_transform, columns=transformed_col_names)

        X_train_non_transform = X_train[self.non_proc_col_names]
        X_pred_non_transform = test_df[self.non_proc_col_names]

        self.train_df = pd.concat([X_train_transform, X_train_non_transform.reset_index(drop=True)], axis=1)
        self.X_pred = pd.concat([X_pred_transform, X_pred_non_transform.reset_index(drop=True)], axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.train_df,
            y_train,
            test_size=0.15
        )
        self.train_df[self.target_col_name] = y_train.reset_index(drop=True)

    def clean_inputs(self, df, training=False):
        # Remove identifier cols (ones that have no direct corr over output)
        df = df.drop(self.identifier_col_names, axis=1)

        df[self.boolean_col_names] = df[self.boolean_col_names].astype(bool)

        if training:
            X_train = df.drop(self.target_col_name, axis=1)
            y_train = df[self.target_col_name]
            return X_train, y_train
        else:
            return df
    
    def create_data_transformer(self):
        numerical_pipeline = make_pipeline(StandardScaler())
        categorical_pipeline = make_pipeline(OneHotEncoder(handle_unknown='ignore'))

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, self.numerical_col_names),
                ('cat', categorical_pipeline, self.one_hot_col_names)
            ]
        )

    # and 3!
    
    @property
    def train_X(self):
        return self.X_train
    
    @property
    def train_y(self):
        return self.y_train
    
    @property
    def test_X(self):
        return self.X_test
    
    @property
    def test_y(self):
        return self.y_test
    
    @property
    def testing_ids(self):
        return self.test_ids
    
    @property
    def submission_set(self):
        return self.X_pred