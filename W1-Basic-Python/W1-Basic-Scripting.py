# Importing packages
import os

from pathlib import Path
import pandas as pd

# Basic python scripting using object-oriented coding
'''
Using the corpus called 100-english-novels, write a Python programme which does the following:

Calculate the total word count for each novel
Calculate the total number of unique words for each novel
Save result as a single file consisting of three columns: filename, total_words, unique_words
''' 

# Defining main function
def main():
    CountFunctions()

# Setting class 'CountFunctions'
class CountFunctions:

    def __init__(self):

        data_dir = self.setting_data_directory()

        files = self.get_paths_from_data_directory(data_dir)

        filenames = self.get_filenames(files)

        total_words = self.get_total_words(files)

        unique_words = self.get_unique_words(files)

        df = self.get_dataframe(filenames=filenames,
                                       total_words=total_words,
                                       unique_words=unique_words)

        dataframe_path = 'cleaned_data.csv' # Setting file name

        df.to_csv(dataframe_path)  # Writing data to csv file

    # Defining function for setting directory for the raw data
    def setting_data_directory(self):

        root_dir = Path.cwd()  # Setting root directory

        data_dir = root_dir / 'data' / '100_english_novels' / 'corpus'  # Setting data directory

        return data_dir
    
    # Defining function for obtaining individual filepaths for data files
    def get_paths_from_data_directory(self, data_dir):
        '''
        Creates a list containing paths to filenames in a data directory
        Args:
            data_dir: Path to the data directory.
        Returns:
            files (list): List of individual file paths
        '''
        files = [] # Creating empty list

        # Loop for iterating through all files in the directory and append individual file paths to the empty list files
        for file in data_dir.glob('*.txt'): 
            files.append(file)

        return files

    # Defining function for obtaining filenames from the filepaths
    def get_filenames(self, files):
        '''
        Creates a list of filenames in a directory.
        Args:
            files (list): List of file paths
        Returns:
            filename: list of filenames
        '''

        filename = []  # Creating empty list

        # Loop for iterating through the different files
        for file in files:

            individual_file_name = os.path.split(file)[-1]  # Extracting last chunk of the splitted filepath as the individual filenames

            filename.append(individual_file_name)  # Append each filename to the list

        return filename

    # Defining function for counting number of words in each file
    def get_total_words(self, files):
        '''
        Gets the total number of words from all the files
        Args:
            files (list): List of individual file paths
        Returns:
            total_words (list): List of total words per file in the input list
        '''

        total_words = []  # Creating empty list

        # Loop for iterating through the different files
        for file in files:

            # Read each file
            with open(file, encoding="utf-8") as f:

                novel = f.read()

                f.close()

            tokens = novel.split()  # Splitting novels into single tokens (basically words) according to whitespace

            total_words.append(len(tokens))  # Counting number of tokens for each file and appending to list 'total_words' 

        return total_words

    # Defining function for counting number of unique words in each file
    def get_unique_words(self, files):
        '''
        Gets the number of unique words from all the files
        Args:
            files (list): List of individual file paths
        Returns:
            unique_words (list): List of number of unique words per file in the input list
        '''

        unique_words = []  # Creating empty list

        # Loop for iterating through the different files
        for file in files:

            # Read each file
            with open(file, encoding="utf-8") as f:

                novel = f.read()

                f.close()

            tokens = novel.split() #  Splitting novels into single tokens (basically words) according to whitespace 

            unique_tokens = set(tokens)  # Creating a set of unique tokens using set()

            unique_words.append(len(unique_tokens))  # Counting number of unique tokens for each file and appending to list 'unique_words' 

        return unique_words

    # Defining short function for creating a pd data frame 
    def get_dataframe(self, filenames, total_words, unique_words):
        '''
        Gets the total number of words from all the files
        Args:
            filenames (list): list of filenames
            total_words (list): List of total words per file in the input list
            unique_words (list): List of number of unique words per file in the input list

        Returns:
            df (pd data frame): pd data frame containing all the cleaned data
        '''

        # Initialising a pd data frame with the required columns and adding the data generated by former functions to these collumns 
        df = pd.DataFrame(data= {'filename': filenames, 'total_words': total_words, 'unique_words': unique_words})

        return df

# Executing main function when script is run
if __name__ == '__main__':
    main()