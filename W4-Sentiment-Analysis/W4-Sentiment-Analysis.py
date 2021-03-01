# Importing packages
import os
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Basic python scripting using object-oriented coding
'''
Calculate the sentiment score for every headline in the data. You can do this using the spaCyTextBlob approach that we covered in class or any other dictionary-based approach in Python.
Create and save a plot of sentiment over time with a 1-week rolling average
Create and save a plot of sentiment over time with a 1-month rolling average
Make sure that you have clear values on the x-axis and that you include the following: a plot title; labels for the x and y axes; and a legend for the plot
Write a short summary (no more than a paragraph) describing what the two plots show. You should mention the following points: 1) What (if any) are the general trends? 2) What (if any) inferences might you draw from them?
'''

# Defining main function 
def main(args):
    filename = args.filename

    Sentiment(filename = filename) # Change argument input to generate new files with different keywords or window sizes

# Setting class 'CountFunctions'
class Sentiment:

    def __init__(self, filename):

        data_dir = self.setting_data_directory() # Setting directory of input data 
        out_dir = self.setting_output_directory() # Setting directory of output plots

        self.filename = filename # Setting filename as the provided filename

        if self.filename is None: # If no filename is specified, use test_data.csv as default file
            self.filename = "test_data.csv"


        df = pd.read_csv(data_dir / f'{self.filename}')  # Read csv file
        df["Polarity"] = self.get_polarity_score(text = df["headline_text"])

        write_path = out_dir / "test.csv" # Path for csv file 

        df.to_csv(write_path)

    # Defining function for setting directory for the raw data
    def setting_data_directory(self):

        root_dir = Path.cwd()  # Setting root directory

        data_dir = root_dir / 'data'   # Setting data directory

        return data_dir


    # Defining function for setting directory for the output
    def setting_output_directory(self):

        root_dir = Path.cwd()  # Setting root directory

        out_dir = root_dir / 'output' # Setting output directory

        return out_dir
    

    # Defining function for calculating polarity
    '''

    '''
    def get_polarity_score(self, text):
        nlp = spacy.load("en_core_web_sm")

        spacy_text_blob = SpacyTextBlob()

        nlp.add_pipe(spacy_text_blob)

        polarity = []

        for headline in nlp.pipe(text):
            polarity_score = headline._.sentiment.polarity 
            polarity.append(polarity_score)

        return polarity


# Executing main function when script is run
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--filename',
                        metavar="Filename",
                        type=str,
                        help='The name of the input data file',
                        required=False)

    main(parser.parse_args())