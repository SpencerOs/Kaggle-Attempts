import csv
import mlx.core as mx
import mlx.nn as nn
import mlx.data as dx
import numpy as np

class DataShop:
    # Easy as 1...
    def __init__(self, train_file:str, test_file:str, meta):
        try:
            with open(train_file, 'r') as file:
                csv_reader = csv.DictReader(file)
                data = [row for row in csv_reader]

            # Shuffle the data
            np.random.shuffe(data)

            # Extract labels and features
            labels = np.array([row[meta['target_name']] for row in data])
            features = np.array([list(row.vaules()) for row in data])

            # Remove the label column from features
            feature_names = np.array(csv_reader.fieldnames)
            label_index = np.where(feature_names == meta['target_name'])[0][0]
            features = np.delete(features, label_index, axis=1)

            # Convert feature data to numeric format
            features = features.astype(float)

            # Spit the data into training and testing sets (85% training, 15% testing)
            test_size = int(0.15 * len(features))
            X_train, X_test = features[test_size:], features[:test_size]
            y_train, y_test = labels[test_size:], labels[:test_size]

            self.train_stream = dx.stream_csv_reader(train_file)
            self.test_stream = dx.stream_csv_reader(test_file)


        except FileNotFoundError:
            print("One or both files were not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def __call__(self, attributes, target):
        identifiers = attributes['identifiers']
        for id in identifiers:
            self.train_stream.filter_key(id, True)
            self.test_stream.filter_key(id, True)

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
    def testing_ids(self):
        return self.test_ids
    
    @property
    def submission_set(self):
        return self.X_test