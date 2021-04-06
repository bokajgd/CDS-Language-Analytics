import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os
import argparse
from pathlib import Path
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
        
        # Setting directory of input data 
        data_dir = self.setting_data_directory() 
        # Setting directory of output plots
        out_dir = self.setting_output_directory() 

        # Loading notes
        if notes_file is None:
            notes = pd.read_csv(data_dir / "NOTEEVENT.csv")
        else: 
            notes = pd.read_csv(data_dir / notes_file)
        # Loading general admission data
        if admissions_file is None:
            admissions = pd.read_csv(data_dir / "ADMISSIONS.csv")
        else: 
            noadmissionstes = pd.read_csv(admissions_file)


        #-#-# PREPROCESSING ADMISSIONS DATA #-#-#


        # Convert to datetime
        admissions.ADMITTIME = pd.to_datetime(admissions.ADMITTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
        admissions.DISCHTIME = pd.to_datetime(admissions.DISCHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
        admissions.DEATHTIME = pd.to_datetime(admissions.DEATHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')

        # Sort by subject ID and admission date
        admissions = admissions.sort_values(['SUBJECT_ID','ADMITTIME'])
        admissions = admissions.reset_index(drop = True)

        # Create collumn containing next admission time (if one exists)
        admissions['NEXT_ADMITTIME'] = admissions.groupby('SUBJECT_ID').ADMITTIME.shift(-1)

        # Create collumn containing next admission type 
        admissions['NEXT_ADMISSION_TYPE'] = admissions.groupby('SUBJECT_ID').ADMISSION_TYPE.shift(-1)

        # Replace values with NaN or NaT if readmissions are planned (Category = 'Elective') 
        rows = admissions.NEXT_ADMISSION_TYPE == 'ELECTIVE'
        admissions.loc[rows,'NEXT_ADMITTIME'] = pd.NaT
        admissions.loc[rows,'NEXT_ADMISSION_TYPE'] = np.NaN

        # It is important that we replace the removed planned admissions with the next unplanned readmission. 
        # Therefore, we backfill the removed values with the values from the next row that contains data about an unplanned readmission

        # Sort by subject ID and admission date just to make sure the order is correct
        admissions = admissions.sort_values(['SUBJECT_ID','ADMITTIME'])
        # Back fill removed values with next row that contains data about an unplanned readmission
        admissions[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']] = admissions.groupby(['SUBJECT_ID'])[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']].fillna(method = 'bfill')

        # Add collumn contain the calculated number of the days until the next admission
        admissions['DAYS_NEXT_ADMIT']=  (admissions.NEXT_ADMITTIME - admissions.DISCHTIME).dt.total_seconds()/(24*60*60)

        # It appears that the reason for the negative values is due to the fact that some of these patients are noted as readmitted before being discharged from their first admission.
        # Quick fix for now is to remove these negative values

        # Removing rows for which value in DAYS_NEXT_ADMIT is negative
        admissions = admissions.drop(admissions[admissions.DAYS_NEXT_ADMIT < 0].index) 

        # Change data type of DAYS_NEXT_ADMIT to float
        admissions['DAYS_NEXT_ADMIT'] = pd.to_numeric(admissions['DAYS_NEXT_ADMIT'])


        #-#-# PREPROCESSING NOTES #-#-#


        # Filtering out discharge summaries
        discharge_sums = notes.loc[notes['CATEGORY'] == 'Discharge summary']

        # Filtering out last note per admission as some admissions have multiple discharge summaries 
        discharge_sums = (discharge_sums.groupby(['SUBJECT_ID','HADM_ID']).nth(-1)).reset_index()


        #-#-# MERGING DATAFRAMES #-#-#


        adm_discharge_sums = pd.merge(admissions[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','DAYS_NEXT_ADMIT', 'NEXT_ADMITTIME','ADMISSION_TYPE','DEATHTIME']],
                                discharge_sums[['SUBJECT_ID','HADM_ID','TEXT']], 
                                on = ['SUBJECT_ID','HADM_ID'],
                                how = 'left')

        # Filtering out all cases of NEWBORN admissions
        adm_discharge_sums = adm_discharge_sums[adm_discharge_sums.ADMISSION_TYPE != 'NEWBORN']

        # Filtering out admissions resulting in patient deaths 
        adm_discharge_sums = adm_discharge_sums[adm_discharge_sums.DEATHTIME.isnull()]

        # Removing admissions with no discharge note
        adm_discharge_sums = adm_discharge_sums.drop(adm_discharge_sums[adm_discharge_sums.TEXT.isnull()].index) 

        # Adding a column with a binary 30-day readmission label which is need for classification model (0 = no readmission, 1 = unplnaned readmission)
        adm_discharge_sums['OUTPUT_LABEL'] = (adm_discharge_sums.DAYS_NEXT_ADMIT < 30).astype('int')


        #-#-# SPLITTING DATA #-#-#


        # Shuffling the data randomly
        adm_discharge_sums = adm_discharge_sums.sample(n = len(adm_discharge_sums), random_state = 42)
        adm_discharge_sums = adm_discharge_sums.reset_index(drop = True)

        # Save 20% of the data as validation and test data 
        valid_test=adm_discharge_sums.sample(frac=0.2,random_state=42)

        # Use the rest of the data as training data
        train=adm_discharge_sums.drop(valid_test.index)

        # Split the 20% subset 50/50 into seperate validation and test sets
        test = valid_test.sample(frac = 0.5, random_state = 42)
        valid = valid_test.drop(test.index)


        #-#-# OVERSAMPLING POSITIVE INSTANCES TO CREATE BALANCED TRAINING DATASET #-#-#


        # Splitting the training data into positive and negative cases
        rows_pos = train.OUTPUT_LABEL == 1
        train_pos = train.loc[rows_pos]
        train_neg = train.loc[~rows_pos]

        # Keeping negative samples equal to 4x number of positive samples
        train_neg = train_neg.sample(n = len(train_pos)*5, random_state = 42,  replace = True)

        # Cocnatenating positive and negative cases and sampling more positive cases until same length as negative cases
        train_balanced = pd.concat([train_neg, train_pos.sample(n = len(train_neg), random_state = 42,  replace = True)], axis = 0)

        # Shuffling the order of training samples 
        train_balanced = train_balanced.sample(n = len(train_balanced), random_state = 42).reset_index(drop = True)


        #-#-# CLEANING TEXT AND TF-IDF VECTORISING NOTES #-#-#


        # Remove new lines ('\n') and carriage returns ('\r')
        train_balanced.TEXT = train_balanced.TEXT.str.replace('\n',' ')
        train_balanced.TEXT = train_balanced.TEXT.str.replace('\r',' ')

        # Remove new lines ('\n') and carriage returns ('\r') for validation data
        valid.TEXT = valid.TEXT.str.replace('\n',' ')
        valid.TEXT = valid.TEXT.str.replace('\r',' ')

        # Remove new lines ('\n') and carriage returns ('\r') for test data
        test.TEXT = test.TEXT.str.replace('\n',' ')
        test.TEXT = test.TEXT.str.replace('\r',' ')

        # Defining TF-IDF vectorizer 
        tfidf_vect = TfidfVectorizer(max_features = 30000, 
                            tokenizer = self.tokenizer_better(), 
                            stop_words = 'english',
                            ngram_range = (1,3), # Include uni-, bi- and trigrams
                            max_df = 0.8)
                       

        # Fit vectorizer to discharge notes
        tfidf_vect.fit(train_balanced.TEXT.values)

        # Transform our notes into numerical matrices
        tfidf_vect_notes = tfidf_vect.transform(train_balanced.TEXT.values)
        tfidf_vect_valid_notes = tfidf_vect.transform(valid.TEXT.values)
        tfidf_vect_test_notes = tfidf_vect.transform(test.TEXT.values)

        # Saving training data vocabulary
        vocab_path = out_dir / 'vocabulary.pkl'
        with open(vocab_path, 'wb') as fw:
            joblib.dump(tfidf_vect.vocabulary_, fw)


        # Saving seperate variables containing classification labels
        train_labels = train_balanced.OUTPUT_LABEL
        valid_labels = valid.OUTPUT_LABEL
        test_labels = test.OUTPUT_LABEL
        # Saving them as csv files
        train_labels.to_csv(out_dir / "train_labels.csv")
        valid_labels.to_csv(out_dir / "valid_labels.csv")
        test_labels.to_csv(out_dir / "test_labels.csv")


        # Turning sparse matrixes into np arrays
        tfidf_vect_notes_array = tfidf_vect_notes.toarray()
        tfidf_vect_notes_valid_array = tfidf_vect_valid_notes.toarray()
        tfidf_vect_notes_test_array = tfidf_vect_test_notes.toarray()

        # Save dense arrays as csv files
        pd.DataFrame(tfidf_vect_notes_array).to_csv(out_dir / "tfidf_vect_notes_array.csv")
        pd.DataFrame(tfidf_vect_notes_valid_array).to_csv(out_dir / "tfidf_vect_notes_valid_array.csv")
        pd.DataFrame(tfidf_vect_notes_test_array).to_csv(out_dir / "tfidf_vect_notes_test_array.csv")
    
    
    #-#-# UTILITY FUNCTIONS #-#-#


    # Defining function for setting directory for the raw data
    def setting_data_directory(self):

        root_dir = Path.cwd()  # Setting root directory

        data_dir = root_dir / 'W6-Network-Analysis' / 'data'   # Setting data directory

        return data_dir


    # Defining function for setting directory for the output
    def setting_output_directory(self):

        root_dir = Path.cwd()  # Setting root directory

        out_dir = root_dir / 'W6-Network-Analysis' / 'output' # Setting output directory

        return out_dir


    # Define a tokenizer function
    def tokenizer_better(self, text):    
        punc_list = string.punctuation+'0123456789'
        t = str.maketrans(dict.fromkeys(punc_list, ''))
        text = text.lower().translate(t)
        tokens = word_tokenize(text)
        return tokens
    
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