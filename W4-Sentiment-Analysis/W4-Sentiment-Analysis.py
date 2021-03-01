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
    filename = args.filename # Adding argument that can be specified in command line

    Sentiment(filename = filename) # Calling main class 

# Setting class 'CountFunctions'
class Sentiment:

    def __init__(self, filename):

        data_dir = self.setting_data_directory() # Setting directory of input data 
        out_dir = self.setting_output_directory() # Setting directory of output plots

        self.filename = filename # Setting filename as the provided filename

        if self.filename is None: # If no filename is specified, use test_data.csv as default file
            self.filename = "test_data.csv"

        df = pd.read_csv(data_dir / f'{self.filename}')  # Read csv file
        df["polarity"] = self.get_polarity_score(text = df["headline_text"]) # Adding a column with calculated polarity scores for each headline

        df_grouped_daily = df.groupby('publish_date', as_index=False)['polarity'].mean() # Grouping headlines and calculating a mean score for each day

        self.create_plot(out_dir = out_dir, data = df_grouped_daily['polarity'].rolling(7).mean(), plot_name = "Weekly_Rolling_Average_Polarity") # Generating and saving plot of 7-day rolling average sentiment score

        self.create_plot(out_dir = out_dir, data = df_grouped_daily['polarity'].rolling(30).mean(), plot_name = "Monthly_Rolling_Average_Polarity") # Generating and saving plot of 30-day rolling average sentiment score


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
    Generates a list containing a polarity score for each input
    Args:
        text: Input text
    Returns:
        polarity (list): List containing polarity scores for each headline

    '''
    def get_polarity_score(self, text):

        nlp = spacy.load("en_core_web_sm") # Initialise spaCy

        spacy_text_blob = SpacyTextBlob() # Initialise spaCyTextBlob 

        nlp.add_pipe(spacy_text_blob) # Add spaCyTextBlob as a new component to our spaCy nlp pipeline

        polarity = [] # Creating empy list for 

        for headline in nlp.pipe(text): # Loop through each headline and calculate polarity score
            polarity_score = headline._.sentiment.polarity 
            polarity.append(float(polarity_score)) # Append score to list

        return polarity


    # Defining function for creating a plot of development in sentiment scores 
    def create_plot(self, out_dir, data, plot_name):

        plt.plot(data, label="Rolling Average Polarity", color = "#3E5B75") # Plot data

        plt.title(plot_name) # Define a title

        plt.legend(loc='upper right') # Define legend position

        plt.xlabel('Date') # Define x-axis label

        plt.ylabel('Polarity') # Define y-axis label

        path = out_dir / f"{plot_name}.png" # Define output path using out_dir

        plt.savefig(path) # Save figure

        plt.close()

# Executing main function when script is run
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', 
                        metavar="Filename",
                        type=str,
                        help='The name of the input data file',
                        required=False)

    main(parser.parse_args())