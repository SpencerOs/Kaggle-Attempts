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
                            target:str=None,
                            test_id:str=None):
        self.identifier_col_names = identifiers
        self.numerical_col_names = numerical
        self.boolean_col_names = booleans
        self.one_hot_col_names = one_hots
        self.target_col_name = target
        self.test_id_col_name = test_id

        self.process_train_and_test()

    def process_train_and_test(self):
        self.X_train, self.y_train = self.clean_inputs(self.train_df, training=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_train,
            self.y_train,
            test_size=0.15
        )
        self.test_ids = self.test_df[self.test_id_col_name]
        self.test_df = self.clean_inputs(self.test_df)

        self.create_data_transformer()
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_test = self.preprocessor.transform(self.X_test)
        self.X_pred = self.preprocessor.transform(self.test_df)

        transformed_col_names = self.preprocessor.get_feature_names_out()

        self.X_train = DataFrame(self.X_train, columns=transformed_col_names)
        self.X_test = DataFrame(self.X_test, columns=transformed_col_names)
        self.X_pred = DataFrame(self.X_pred, columns=transformed_col_names)

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
    def training_split_X(self):
        return self.X_train
    
    @property
    def training_split_y(self):
        return self.y_train
    
    @property
    def testing_split_X(self):
        return self.X_test
    
    @property
    def testing_split_y(self):
        return self.y_test
    
    @property
    def train_frame(self):
        return self.train_df
    
    @property
    def testing_ids(self):
        return self.test_ids
    
    @property
    def test_set(self):
        return self.X_pred