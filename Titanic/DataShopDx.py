import csv
import mlx.core as mx
import mlx.nn as nn
import mlx.data as dx
import numpy as np

class DataShopDx:
    # Easy as 1...
    def __init__(self, train_file:str, test_file:str, meta):
        # try:
        with open(train_file, 'r') as file:
            csv_reader = csv.DictReader(file)
            data = [row for row in csv_reader]

        with open(test_file, 'r') as file:
            test_csv_reader = csv.DictReader(file)
            test_data = [row for row in test_csv_reader]

        # Shuffle the data
        np.random.shuffle(data)

        # Extract labels and features
        labels = np.array([row[meta['target_name']] for row in data])
        features = np.array([list(row.values()) for row in data])
        sub_features = np.array([list(row.values()) for row in test_data])

        # Remove the label column from features
        feature_names = np.array(csv_reader.fieldnames)
        label_index = np.where(np.logical_or(feature_names == meta['target_name'], feature_names == meta['identifiers']))[0]
        features = np.delete(features, label_index, axis=1)
        sub_feature_names = np.array(test_csv_reader.fieldnames)
        sub_index = np.where(sub_feature_names == meta['identifiers'])[0][0]
        sub_features = np.delete(sub_features, sub_index, axis=1)

        # Convert feature data to numeric format
        features = features.astype(float)
        sub_features = sub_features.astype(float)

        # Split the data into training and testing sets (85% training, 15% testing)
        test_size = int(0.15 * len(features))
        self.X_train, self.X_test = features[test_size:], features[:test_size]
        self.y_train, self.y_test = labels[test_size:], labels[:test_size]
        self.testing_ids = sub_index
        self.X_val = sub_features

        # except FileNotFoundError:
        #     print("One or both files were not found.")
        # except Exception as e:
        #     print(f"An error occurred: {e}")

    def batch_iterate(self, batch_size):
        perm = mx.array(np.random.permutation(self.y_train.size))
        for s in range(0, self.y_train.size, batch_size):
            ids = perm[s : s + batch_size]
            yield self.X_train[ids], self.y_train[ids]
    
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
        return self.X_val