# Importing packages
import os
import re

from pathlib import Path
import pandas as pd
import numpy as np

# Basic python scripting using object-oriented coding
'''
Using the corpus called 100-english-novels, write a Python programme which does the following:

- The script should take a directory of text files, a keyword, and a window size (number of words) as input parameters, and an output file called out/{filename}.csv
These parameters can be defined in the script itself
- Find out how often each word collocates with the target across the corpus
- Use this to calculate mutual information between the target word and all collocates across the corpus
- Save result as a single file consisting of three columns: collocate, raw_frequency, MI
''' 

'''
Shortcomings of this script:
- The current function concatenates all text files before calculating relevant scores. Therefore, the 'window' slides across the boundaries of each text which is not optimal.
- What if a collocate appears multiple times in a given window? 
'''

# Defining main function 
def main():
    Collocation(keyword = "cat", window_size = 2) # Change argument input to generate new files with different keywords or window sizes

# Setting class 'CountFunctions'
class Collocation:


    def __init__(self, keyword, window_size):
        self.keyword = keyword # Defining keyword as the parsed argument 'keyword'
        self.window_size = window_size # Defining keyword as the parsed argument 'window_size'

        data_dir = self.setting_data_directory() # Setting data directory 
        out_dir = self.setting_output_directory() # Setting output directory for the generated csv file
        files = self.get_paths_from_data_directory(data_dir) # Getting list of filepaths for the images

        tokenized_text = self.get_tokenized_concatenated_texts(files) # Getting tokenized version off concatenated text corpus

        collocates, R1 = self.get_list_of_collocates(tokenized_text, self.keyword , self.window_size)  # Getting list of unique collocates

        raw_frequencies = [] # Empty list for raw frequencies of collocates
 
        MIs = [] # Empty list for MI-scores between keyword and collocate for all collocates

        # Loop through collocate
        for collocate in collocates: 

            collocate_raw_frequency = self.get_raw_frequency(tokenized_text, collocate)  # Raw frequency of collocate

            O11 = self.get_O11(tokenized_text, self.keyword , collocate, self.window_size)  # Joint frequency of keyword and collocate

            C1 = collocate_raw_frequency  # Calculating: Same as raw frequency of collocate

            N = len(tokenized_text)  # O11 + O12 + O21 + O22

            E11 = (R1 * C1 / N)  # Expected frequency

            MI = np.log2(O11 / E11)  # Mutual information

            # Adding information for given collocate to list
            raw_frequencies.append(collocate_raw_frequency) 
            MIs.append(MI)

        # Gathering needed lists into a dictionary
        data_dict = {"collocate": collocates, 
                     "raw_frequency": raw_frequencies,
                     "MI": MIs}

        # Creating pd data frame from dictionary
        df = pd.DataFrame(data=data_dict)

        df = df.sort_values("MI", ascending=False)  # Sorting collocates with highest frequency at the top.

        write_path = out_dir / f"{self.keyword}_collocates_ws_{self.window_size}.csv" # Path for csv file 

        df.to_csv(write_path) # Writing csv files



    # Defining function for setting directory for the raw data
    def setting_data_directory(self):

        root_dir = Path.cwd()  # Setting root directory

        data_dir = root_dir / 'data' / '100_english_novels' / 'corpus'  # Setting data directory

        return data_dir


    # Defining function for setting directory for the output
    def setting_output_directory(self):

        root_dir = Path.cwd()  # Setting root directory

        out_dir = root_dir / 'output' # Setting output directory

        return out_dir
    

    # Defining function for obtaining individual filepaths for data files
    '''
    Creates a list containing paths to filenames in a data directory
    Args:
        data_dir: Path to the data directory.
    Returns:
        files (list): List of individual file paths
    '''
    def get_paths_from_data_directory(self, data_dir):

        files = [] # Creating empty list

        # Loop for iterating through all files in the directory and append individual file paths to the empty list files
        for file in data_dir.glob('*.txt'): 
            files.append(file)

        return files


    # Defining function for finding all unique collocate
    '''
    Creates a list of unique collocates for the specific keyword
    Args:
        tokenized_text (string): Tokenised version of full text corpus
        keyword (string): Specified keyword
        collocate (string): Specified collocate of the keyword
        window_size (integer): Specified window size in which to search for collocates
    Returns:
        unique_collocates (list): List of unique collocates
    '''
    def get_list_of_collocates(self, tokenized_text, keyword, window_size):

            collocates = [] # Create empty list for all collocate

            for i, word in enumerate(tokenized_text):

                if word == keyword:

                    left_window = tokenized_text[max(0, i - window_size):i]

                    right_window = tokenized_text[i+1:(i + window_size)]

                    total_window = left_window + right_window

                    collocates.extend(total_window) # Add all collocates from window to the full list of collocates

            unique_collocates = pd.unique(collocates) # Only keep one of each unique collocate
            R1 = len(collocates) # Number of all collocates in each window combined

            return unique_collocates, R1


    # Defining function for counting raw frequency of a word in the tokensied text corpus
    '''
    Generates a count of the word frequency in the coprus for a given word
    Args:
        tokenized_text (list): Tokenised version of full text corpus
    Returns: 
        word_frequency (integer): Count of word frequency
    '''
    def get_raw_frequency(self, tokenized_text, focus_word):

        word_frequency = 0

        for word in tokenized_text:

            if word == focus_word:

                word_frequency += 1

        return word_frequency
    

    # Defining function for getting frequency of keyword and collocate appearing together
    '''
    Creates a count of the  frequency of keyword and collocate appearing together
    Args:
        tokenized_text (string): Tokenised version of full text corpus
        keyword (string): Specified keyword
        collocate (string): Specified collocate of the keyword
        window_size (integer): Specified window size in which to search for collocates
    Returns:
        O11 (integer): O11 value for the specified keyword and collocate
    '''
    def get_O11(self, tokenized_text, keyword, collocate, window_size):

        O11 = 0

        for i, word in enumerate(tokenized_text):

            if word == keyword: # Record all instances of keyword and create a window around it

                left_window = tokenized_text[max(0, i - window_size):i]

                right_window = tokenized_text[i:(i + window_size + 1)]

                total_window = left_window + right_window

                if keyword and collocate in total_window: # If collocate is in the window then add one to count

                    O11 += 1

        return O11


    # Defining function for opening and reading text files
    '''
    Loads and reads a txt file
    Args:
        file: A path to an image file.
    Returns:
       text (string): Text of a txt file
    '''
    def load_text(self, file):
        # Read each .txt file
        with open(file, encoding="utf-8") as file:

            text = file.read()
            
            text = text.lower() # Using lowercase method to set all characters to lowercase

            file.close()

        return text


    # Defining tokeniser function
    '''
    Tokenizes an input text:
    Args:
        input_text (string): Input text
    Returns:
        tokenized_text (list): List of individual tokens
    '''
    def tokenize(self, input_text):
        # Split on any non-alphanumeric character
        tokenizer = re.compile(r"\W+")

        # Tokenize
        tokenized_text = tokenizer.split(input_text) 

        # Return token_list
        return tokenized_text


    # Defining function for concatenating .txt files into one text    
    '''
    Creates a combined text from multiple files
    Args:
        files (list): List of file paths
    Returns:
        text_corpus (string): Full tokenised text
    '''
    def get_tokenized_concatenated_texts(self, files):
        tokenized_text_corpus = []

        for file in files:

            text = self.load_text(file)

            tokenized_text = self.tokenize(text)

            tokenized_text_corpus.extend(tokenized_text)

        return tokenized_text_corpus


# Executing main function when script is run
if __name__ == '__main__':
    main()