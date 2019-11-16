"""Exercise 1

Usage:

$ CUDA_VISIBLE_DEVICES=2 python practico_1_train_petfinder.py --dataset_dir ../ --epochs 30 --dropout 0.1 0.1 --hidden_layer_sizes 200 100

To know which GPU to use, you can check it with the command

$ nvidia-smi
"""

import argparse

import os
import mlflow
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models


import warnings
warnings.filterwarnings("ignore")

from auxiliary import process_features, load_dataset, build_columns, log_dir_name

TARGET_COL = 'AdoptionSpeed'


def read_args():
    parser = argparse.ArgumentParser(
        description='Training a MLP on the petfinder dataset')
    # Here you have some examples of classifier parameters. You can add
    # more arguments or change these if you need to.
    parser.add_argument('--experiment_name', type=str, default='Base model',
                        help='Name of the experiment, used in mlflow.')
    parser.add_argument('--dataset_dir', default='../petfinder_dataset', type=str,
                        help='Directory with the training and test files.')
    parser.add_argument('--hidden_layer_sizes', nargs='+', default=[100], type=int,
                        help='Number of hidden units of each hidden layer.')
    parser.add_argument('--epochs', default=50, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--dropout', nargs='+', default=[0.5], type=float,
                        help='Dropout ratio for every layer.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of instances in each batch.')
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        help='Learning rate.')
    args = parser.parse_args()

    assert len(args.hidden_layer_sizes) == len(args.dropout)
    return args

def print_args(args):
    print('-------------------------------------------')
    print('PARAMS ------------------------------------')
    print('-------------------------------------------')
    print('--experiment_name   ', args.experiment_name)
    print('--dataset_dir       ', args.dataset_dir)
    print('--epochs            ', args.epochs)
    print('--hidden_layer_sizes', args.hidden_layer_sizes)
    print('--dropout           ', args.dropout)
    print('--batch_size        ', args.batch_size)
    print('--learning_rate     ', args.learning_rate)
    print('-------------------------------------------')


def main():
    args = read_args()
    print_args(args)

    experiment_name = args.experiment_name
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    hidden_layer_sizes = args.hidden_layer_sizes
    dropout = args.dropout
    epochs = args.epochs

    ### Output directory
    dir_name = log_dir_name(args)
    print()
    print(dir_name)
    print()
    output_dir = os.path.join('experiments', experiment_name, dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dataset, dev_dataset, test_dataset = load_dataset(args.dataset_dir)
    nlabels = dataset[TARGET_COL].unique().shape[0]

    columns = ['Gender', 'Color1']
    one_hot_columns, embedded_columns, numeric_columns = build_columns(dataset, columns)

    # TODO (optional) put these three types of columns in the same dictionary with "column types"
    X_train, y_train = process_features(dataset, one_hot_columns, numeric_columns, embedded_columns)
    direct_features_input_shape = (X_train['direct_features'].shape[1],)
    X_dev, y_dev = process_features(dev_dataset, one_hot_columns, numeric_columns, embedded_columns)
      

    ###########################################################################################################
    ### TODO: Shuffle train dataset - Done
    ###########################################################################################################
    shuffle_len = X_train['direct_features'].shape[0]
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(shuffle_len).batch(batch_size)
    ###########################################################################################################

    dev_ds = tf.data.Dataset.from_tensor_slices((X_dev, y_dev)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices(process_features(
        test_dataset, one_hot_columns, numeric_columns, embedded_columns, test=True)[0]).batch(batch_size)

    ###########################################################################################################
    ### TODO: Build the Keras model - Done
    ###########################################################################################################
    tf.keras.backend.clear_session()

    # Add the direct features already calculated
    direct_features_input = layers.Input(shape=direct_features_input_shape, name='direct_features')
    inputs = [direct_features_input]
        
    features = direct_features_input

    denses = []
    dense1 = layers.Dense(hidden_layer_sizes[0], activation='relu')(features)
    denses.append(dense1)
    if len(hidden_layer_sizes) > 1:
        for hidden_layer_size in hidden_layer_sizes[1:]:
            dense = layers.Dense(hidden_layer_size, activation='relu')(denses[-1])
            denses.append(dense)
    output_layer = layers.Dense(nlabels, activation='softmax')(dense1)

    model = models.Model(inputs=inputs, outputs=output_layer)
    ###########################################################################################################
    

    ###########################################################################################################
    ### TODO: Fit the model - Done
    ###########################################################################################################
    mlflow.set_experiment(experiment_name)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
              metrics=['accuracy'])

    logdir = "logs/scalars/" + dir_name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir)

    with mlflow.start_run(nested=True):
        # Log model hiperparameters first
        mlflow.log_param('hidden_layer_size', hidden_layer_sizes)
        mlflow.log_param('dropout', dropout)
        mlflow.log_param('embedded_columns', embedded_columns)
        mlflow.log_param('one_hot_columns', one_hot_columns)
        mlflow.log_param('numeric_columns', numeric_columns)  # Not using these yet
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('learning_rate', learning_rate)
        

        # Train
        history = model.fit(train_ds, epochs=epochs,
                            validation_data=dev_ds,
                            callbacks=[tensorboard_callback])

        #######################################################################################################
        ### TODO: analyze history to see if model converges/overfits
        #######################################################################################################
        output_csv = os.path.join(output_dir, 'history.pickle')
        with open(output_csv, 'bw') as f:
            pickle.dump(history.history, f)
        #######################################################################################################


        #######################################################################################################
        ### TODO: Evaluate the model, calculating the metrics. - Done
        #######################################################################################################
        loss, accuracy = model.evaluate(dev_ds)
        print("*** Dev loss: {} - accuracy: {}".format(loss, accuracy))
        mlflow.log_metric('loss', loss)
        mlflow.log_metric('accuracy', accuracy)       
        predictions = model.predict(test_ds)
        #######################################################################################################


        #######################################################################################################
        ### TODO: Convert predictions to classes - Done
        #######################################################################################################
        prediction_classes = np.argmax(predictions, axis=1)
        #######################################################################################################


        #######################################################################################################
        ### TODO: Save the results for submission - Done
        #######################################################################################################
        output_csv = os.path.join(output_dir, 'submit.csv')
        submissions = pd.DataFrame(prediction_classes, columns=[TARGET_COL], index=test_dataset.PID)
        submissions.to_csv(output_csv)
        #######################################################################################################


   ###########################################################################################################

print('All operations completed')

if __name__ == '__main__':
    main()
