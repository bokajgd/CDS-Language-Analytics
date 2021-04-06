import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import joblib

import scipy.sparse

import string

import nltk
from nltk import word_tokenize
nltk.download('punkt')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# **Preprocessing and preperation of data**

# The purpose of this script is to prepare and preproces the raw textual data and the admission data needed for training and testing the classification model. This proces includes the following steps:

# 1. Clean and prepare admission data
# 2. Extract discharge summaries from note data
# 3. Remove newborn cases and in-hospital deaths
# 4. Bind note-data to 30-day readmission information
# 5. Split into train, validation and test set and balance training data by oversampling positive cases
# 6. Removal of special characters, numbers and de-identified brackets
# 7. Vectorise all discharge notes:
#   7a.  Remove stop-words, most common words and very rare words (benchmarks need to be defined)
#   7b. Create set of TF-IDF weighted tokenised discharge notes
# 8. Output datasets and labels as CSV-files


# Defining main function
def main(args):
    notes_file = args.nf 
    admissions_file = args.af

    NotePreprocessing(notes_file = notes_file, admissions_file = admissions_file)

# Defining class 'NotePreprocessing'
class NotePreprocessing:
    def __init__(self, notes_file, admissions_file):
        # Loading notes
        if notes_file is None:
            notes = pd.read_csv("NOTEEVENT.csv")
        else: 
            notes = pd.read_csv(notes_file)
        # Loading general admission data
        if admissions_file is None:
            admissions = pd.read_csv("ADMISSIONS.csv")
        else: 
            noadmissionstes = pd.read_csv(admissions_file)



# Executing main function when script is run
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--nf', 
                        metavar="Notes file",
                        type=str,
                        help='The path for the file containing clinical notes',
                        required=False)

    parser.add_argument('--af', 
                        metavar="Admissions file",
                        type=str,
                        help='The path for the file containing general admissions dara',
                        required=False)

    main(parser.parse_args())