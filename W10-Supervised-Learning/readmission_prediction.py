import tensorflow as tf
import keras
from keras.layers import Input, Dense, LeakyReLU, InputLayer
from keras.datasets import mnist
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keract import get_activations, display_activations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import kerastuner
from kerastuner import HyperModel, Objective, Hyperband, BayesianOptimization

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score

# Define leaky lrelu
lrelu = lambda x: tf.keras.layers.LeakyReLU(alpha=0.3)(x)

'''
This script trains a logistic regression classifier on the cleaned tf-idf vectorised notes 
producted by the data_preprocessing.py script and tests it's ability to predict unplanned readmissions
'''

# Defining main function
def main(args):
    train_data = args.train
    test_data = args.test

    train_labels = args.trl
    test_labels = args.tel

    ReadmissionPrediction(train_data = train_data, 
                      test_data = test_data,
                      train_labels = train_labels,
                      test_labels = test_labels)

# Defining class 'ReadmissionPrediction'
class ReadmissionPrediction:
    def __init__(self, train_data, test_data, train_labels, test_labels):
        
        # Setting directory of input data 
        data_dir = self.setting_data_directory() 
        # Setting directory of output plots
        out_dir = self.setting_output_directory() 

        # Loading training data
        if train_data is None:
            train_data = self.load_data(file_name = 'tfidf_vect_notes_array') 
        else:
            train_data = self.load_data(file_name = train_data) 
        # Loading test data 
        if test_data is None:
            test_data = self.load_data(file_name = 'tfidf_vect_notes_test_array')
        else:
            test_data = self.load_data(file_name = test_data) 
        # Load training labels
        if train_labels is None:
            train_labels = self.load_data(file_name = 'train_labels')
        else:
            train_labels = self.load_data(file_name = train_labels)
        # Load training labels
        if test_labels is None:
            test_labels = self.load_data(file_name = 'test_labels')
        else:
            test_labels = self.load_data(file_name = test_labels)


        #-#-# LOGISTIC REGRESSION #-#-#


        # Defining logistic regression using keras functional api
        input_baseline = Input(shape = tfidf_input_dim)
        lr_baseline = Dense(classification_dim, activation = 'sigmoid'  , kernel_initializer='he_normal')(input_baseline)
        baseline = Model(input_baseline, lr_baseline , name = 'baseline') 

        # Compiling network
        baseline.compile(optimizer=Adam(learning_rate = 0.0001), loss='binary_crossentropy', metrics=['accuracy', 'AUC', 'Precision', 'Recall', tf.keras.metrics.PrecisionAtRecall(recall=0.8), AUC_PR])

        # Fitting to noisy input data
        baseline.fit(train_data, train_labels, epochs=15, batch_size=25, shuffle = True)

        # Evaluate model
        evaluation = baseline.evaluate(test_data, test_labels, verbose=1) 
        print(evaluation)


        #-#-# AUC-ROC CURVE #-#-#


        # Calculatinng false positive and true positive rates
        baseline_preds = best_model.predict(test_data).ravel()
        fpr, tpr, thresholds = roc_curve(test_labels, baseline_preds)

        # Calculating AUC for ROC curve
        auc_roc = auc(fpr, tpr)
        # Plot curve
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='Logistic regression classifier (area = {:.3f})'.format(auc_roc))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig(Path.cwd() / 'W10-Supervised-Learning' / 'output' / 'auc_roc.png')


        #-#-# UTILITY FUNCTIONS #-#-#


        # Load data function
        def load_data(self, file_name):
            data = pd.read_csv(Path.cwd() / 'W10-Supervised-Learning' / 'data' / f'{file_name}.csv') # Load data
            data = data.drop(data.columns[0], axis=1) # Remove first column
            data = data.to_numpy() # Convert to np array
            return data


# Executing main function when script is run
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', 
                        metavar="Training Data",
                        type=str,
                        help='The file name of the training data csv',
                        required=False)

    parser.add_argument('--test', 
                        metavar="Test Data",
                        type=str,
                        help='The file name of the test data csv,
                        required=False)

    parser.add_argument('--trl', 
                        metavar="Training Labels",
                        type=str,
                        help='The file name of the training labels csv',
                        required=False)
                        
    parser.add_argument('--tel', 
                        metavar="Test Labels",
                        type=str,
                        help='The file name of the test labels csv',
                        required=False)


    main(parser.parse_args())