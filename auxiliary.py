import os
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

TARGET_COL = 'AdoptionSpeed'

def log_dir_name(args):
    dir_name = 'batch_size__{}'.format(args.batch_size)

    dir_name += '__epochs__{}'.format(args.epochs)

    dir_name += '__learning_rate__{}'.format(args.learning_rate)

    dir_name += '__hidden_layer_sizes__'
    dir_name += '_'.join(map(str, args.hidden_layer_sizes))

    dir_name += '__dropout__'
    dir_name += '_'.join(map(str, args.dropout))

    return dir_name

def process_features(df, one_hot_columns, numeric_columns, embedded_columns, test=False):
    direct_features = []

    # Create one hot encodings
    for one_hot_col, max_value in one_hot_columns.items():
        direct_features.append(tf.keras.utils.to_categorical(df[one_hot_col] - 1, max_value))

    ###########################################################################################################
    ### TODO Create and append numeric columns - Done
    ###########################################################################################################

    for numeric_col in numeric_columns:
        
        values = df[numeric_col].values
        scaled_col = (values - values.min()) / (values.max() - values.min())
        direct_features.append(scaled_col.reshape(-1,1))

    ###########################################################################################################
    
    # Concatenate all features that don't need further embedding into a single matrix.
    features = {'direct_features': np.hstack(direct_features)}

    # Create embedding columns - nothing to do here. We will use the zero embedding for OOV
    for embedded_col in embedded_columns.keys():
        features[embedded_col] = df[embedded_col].values

    if not test:
        nlabels = df[TARGET_COL].unique().shape[0]
        # Convert labels to one-hot encodings
        targets = tf.keras.utils.to_categorical(df[TARGET_COL], nlabels)
    else:
        targets = None
    
    return features, targets

def load_dataset(dataset_dir):

    # Read train dataset (and maybe dev, if you need to...)
    dataset, dev_dataset = train_test_split(
        pd.read_csv(os.path.join(dataset_dir, 'train.csv')), test_size=0.2)
    
    test_dataset = pd.read_csv(os.path.join(dataset_dir, 'test.csv'))
    
    print('Training samples {}, test_samples {}'.format(
        dataset.shape[0], test_dataset.shape[0]))
    
    return dataset, dev_dataset, test_dataset



def build_columns(dataset, columns):

    columns = set(columns)
    one_hot_cols  = columns.intersection(set(['Gender', 'Color1', 'Vaccinated', 'Dewormed', 'Health']))
    embedded_cols = columns.intersection(set(['Breed1']))
    numeric_cols  = columns.intersection(set(['Age', 'Fee', 'Quantity']))

    one_hot_columns = {
        one_hot_col: dataset[one_hot_col].max()
        for one_hot_col in one_hot_cols
    }
    embedded_columns = {
        embedded_col: dataset[embedded_col].max() + 1
        for embedded_col in embedded_cols
    }
    numeric_columns = list(numeric_cols)

    return one_hot_columns, embedded_columns, numeric_columns