
# Load packages
import os
from pathlib import Path
import pandas as pd
import re 
import string
import numpy as np


#DEFINE PARAMETERS; file_path, keyword, window_size = 50
keyword = "house"
window_size = 30
file_path = os.path.join("..","data","100_english_novels", "corpus")


# Function to calculate word frequency from a list of words
def freq(wordlist): 
  
    # Create empty 'result list'
    result = []
    
    # break the string into list of words           
    new_string = [] 
  
    # loop over words present in wordlist
    for word in wordlist:              
  
        # checking for the duplicacy 
        if word not in new_string: 
  
            # insert value in new_string
            new_string.append(word)  
              
    for i in range(0, len(new_string)): 
  
        # Append the list of words and frequencies to results
        result.append((new_string[i],wordlist.count(new_string[i])))

    return(result)

# Function to sort the word frequency pairs
def word_sorter(x):  
      
    # getting length of list of word/frequency pairs 
    lst = len(x)  

    # sort by frequency
    for i in range(0, lst):  
          
        for j in range(0, lst-i-1):  
            if (x[j][1] > x[j + 1][1]):  
                temp = x[j]  
                x[j]= x[j + 1]  
                x[j + 1] = temp  
    return x

# Function to tokenize strings of text
def tokenize (library):

    # Make sure all words are written with capital letters
    library = library.upper()

    # Remove non-alphabetic characters
    tokenizer = re.compile(r"\W+")

    # tokenize
    tokenized_library = tokenizer.split(library)

    return(tokenized_library)

# Function to find collocates, frequencies, and MI
def main(keyword, window_size, file_path):

    # Create a logfilename given the keyword
    logfilename = 'out/collocates_keyword_{}.csv'.format(keyword)

    corpus = ""
    for filename in Path(file_path).glob("*86.txt"): # Create a subset of the novels (only from 1886 and 1986)
    
        # Read the given text file
        with open (filename, "r", encoding = "utf-8") as file:
            loaded_text = file.read()

            corpus = corpus + loaded_text


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

    # count frequency of each unique word in the window library and save 
    # the frequencies with the word in a list. The list is sorted by frequency
    tokenized_library = tokenize(window_library)
    word_freq_pairs = word_sorter(freq(tokenized_library)) 

    # Tokenize the corpus and make sure the keyword is written with capital letters
    tokenized_corpus = tokenize(corpus)
    keyword = keyword.upper()

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
    
        # C1 will be zero in case a word does not exist in the corpus 
        # because it was split in the the middle when extracting windows
        # Calculate MI for actual words
        if not (C1 == 0):
            
            # expected co-occurance frequency
            E11 = (R1*C1/N)
            if (E11 > 0):
                # calculate MI score
                MI = np.log(O11/E11)

                print(f"word: {collocate}, MI = {MI}, O11: {O11}, O12: {O12}, R1: {R1}, O21: {O21}, C1 = {C1}, N = {N}, E11 is {E11}")


                # Wrap up in DF
                DATA = DATA.append({
                    'collocate': collocate,
                    'raw_frequency': O11,
                    'MI': MI,
                    }, ignore_index=True) 
        
    # Save as a csv file
    DATA.to_csv(logfilename)
    print("Done!")


# Define behaviour when called from command line
if __name__=="__main__":
    main(keyword, window_size, file_path)