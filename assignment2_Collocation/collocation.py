#!/usr/bin/env python

#_______________# Import packages #_______________# 
import os
from pathlib import Path
import pandas as pd
import re 
import string
import numpy as np
import argparse

#_______________# The script #_______________#

### FUNCTIONS

# frequency function
def freq(wordlist): 
    """
    Function to calculate word frequency from a list of words.
    It returns the list of words together with frequencies.
    """
    # Create empty 'result list'
    result = []
    # break the string into list of words           
    new_string = [] 
    
    # loop over words present in wordlist
    for word in wordlist:              
        # checking for duplicates
        if word not in new_string: 
            # insert value in new_string
            new_string.append(word)  
              
    for i in range(0, len(new_string)): 
        # Append the list of words and frequencies to results
        result.append((new_string[i],wordlist.count(new_string[i])))

    return(result)

# word sorter function
def word_sorter(x):  
    """
    Function to sort the word frequency pairs after frequency
    Lowest frequency collocates first - highest frerquency collocates last
    """
    # getting length of list of word/frequency pairs 
    lst = len(x)  

    # sort by frequency
    for i in range(0, lst):    
        for j in range(0, lst-i-1):  
            if (x[j][1] > x[j + 1][1]):  
                temp = x[j]  
                x[j]= x[j + 1]  
                x[j + 1] = temp  
    return(x)

# Tokenize function
def tokenize(library):
    """
    Function that takes a string of text a tokenizes it.
    The text is returned tokenized.
    """
    # Make sure all words are written with lower cases
    library = library.lower()
    # Remove non-alphabetic characters
    tokenizer = re.compile(r"\W+")
    # tokenize
    tokenized_library = tokenizer.split(library)

    return(tokenized_library)


# text in window function
def text_in_windows(keyword, corpus, window_size):
    """
    Function that takes a keyword, the corpus, and a window size and gathers all text 
    appearing within the windows around the keyword appearing in the corpus.
    The function returns the window library.
    """
    # Creating a library of all text co-occuring with the keyword
    window_library = ""

    for match in re.finditer(keyword, corpus):
        
        # first character index of match
        word_start = match.start()
        # last character index of match
        word_end = match.end()
        
        # Left window
        left_window_start = max(0, word_start-window_size)
        left_window = corpus[left_window_start:word_start]
        
        # Right window
        right_window_end = word_end + window_size
        right_window = corpus[word_end : right_window_end]

        # Save right/left windows in the window library
        window_library = window_library + left_window + right_window
    
    return(window_library)


### THE MAIN FUNCTION

# Function to find collocates, frequencies, and MI
def main(args):
    
    # Import parameters specified in the terminal
    keyword = args["keyword"]
    window_size = args["window_size"]
    folder = args["file_path"]
    file_path = "data/" + folder + "/"

    # make sure the keyword is written with lower cases.
    keyword = keyword.lower()
    
    # Collect all texts from the data folder in one long string called 'corpus'.
    corpus = ""
    for filename in Path(file_path).glob("*.txt"): # Take all txt files in the data folder
        
        # Read the given text file
        with open (filename, "r", encoding = "utf-8") as file:
            loaded_text = file.read()
            corpus = corpus + loaded_text # append text to corpus.
            
            
    # Collect all text appearing within the windows of the keyword in the corpus.
    window_library = text_in_windows(keyword, corpus, window_size)
    
    # Count frequency of each unique word in the window library and save the frequencies with the word in a list. 
    # The list is sorted by frequency
    tokenized_library = tokenize(window_library)
    word_freq_pairs = word_sorter(freq(tokenized_library)) 

    # Tokenize the corpus
    tokenized_corpus = tokenize(corpus)

    # Create pandas to store the results for each word 
    Columns = ['collocate','raw_frequency','MI']           
    DATA = pd.DataFrame(columns = Columns)

    # Gather all information about each word, calculate MI, and create a row per word in the DF
    for word in word_freq_pairs:
        # the collocate
        collocate = word[0]
        # the co-occurance frequency
        O11 = word[1]
        # keyword general appearance in the corpus
        R1 = tokenized_corpus.count(keyword)
        # keyword appearance without collocate
        O12 = R1 - O11
        # collocate general appearance
        C1 = tokenized_corpus.count(collocate)
        # collocate appearance without keyword
        O21 = C1 - O11
        # total number of words in the corpus
        N = len(tokenized_corpus)
        """
        C1 will be zero in case a word does not exist in the corpus 
        because it was split in the the middle when extracting windows
        Calculate MI for actual words
        """
        if not (C1 == 0):
            
            # expected co-occurance frequency
            E11 = (R1*C1/N)
            if (E11 > 0):
                # calculate MI score
                MI = np.log(O11/E11)

                # Wrap up in DF
                DATA = DATA.append({
                    'collocate': collocate,
                    'raw_frequency': O11,
                    'MI': MI,
                    }, ignore_index=True) 
        
        
    # Create a logfilename given the keyword and save the df as a csv
    logfilename = 'collocates_keyword_{}.csv'.format(keyword)
    DATA.to_csv(f"out/{logfilename}")
    print(f"A csv-file '{logfilename}' with all collocates has been saved in the folder 'out'.")



# Define behaviour when called from command line
if __name__=="__main__":
    
    # Argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--keyword", type = str, required = True,
                    help = "Specify a keywords to which you wish to calculate collocations.")
    ap.add_argument("-w", "--window_size", type = int, default = 30,
                    help = "Specify a window size of co-occurance. The default is 30.")
    ap.add_argument("-f", "--file_path", type = str, default = "Stevenson",
                    help = "Specify a folder in 'data' with the files in which you wish to find collocations. The default is 'Stevenson'.")

    # Parse arguments. args is now an object containing all arguments added through the terminal. 
    argument_parser = vars(ap.parse_args())
    
    # run main() function
    main(argument_parser)
    