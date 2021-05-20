#!/usr/bin/env python

# _________ Install packages _________#

# system tools
import os
import sys
import argparse
sys.path.append(os.path.join(".."))

# data munging tools
import pandas as pd
import numpy as np
from itertools import repeat
import itertools

# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# _________ The script _________#



# DEFINE FUNCTIONS

# heatmap, confusion matrix function
def plot_cm(y_test, y_pred, filename, normalized:bool):
    """
    This function is retrieved from Ross' repository and only slightly modified
    https://github.com/CDS-AU-DK/cds-language/blob/main/utils/classifier_utils.py
    
    The Function will create and save a confusion matrix illustrating how the model performs.
    """
    if normalized == False:
        cm = pd.crosstab(y_test, y_pred, 
                            rownames=['Actual'], colnames=['Predicted'])
        p = plt.figure(figsize=(10,10));
        p = sns.heatmap(cm, annot=True, fmt="d", cbar=False)
        plt.savefig(filename)
    
    elif normalized == True:
        cm = pd.crosstab(y_test, y_pred, 
                               rownames=['Actual'], colnames=['Predicted'], normalize='index')
        p = plt.figure(figsize=(10,10));
        p = sns.heatmap(cm, annot=True, fmt=".2f", cbar=False)
        plt.savefig(filename)

        return None


# Chunk function
def chunk_me(DATA, chunk_size):
    """
    Function that will take the data and return two lists, 
    one with labels and one with corresponding text chunks given the chunk size defined
    """
    
    # Get a list of uniques seasons (1 to 8)
    seasons = DATA.Season.unique()

    # Create empty list to save text chunks and corresponding labels
    text = []
    labels = []

    # Loop over every season and chunk lines together + save season label.
    for i in seasons:
        
        # Get data from season
        season = DATA[DATA["Season"] == i]  
        
        # Get lines
        sentences = season["Sentence"]
        
        # Chunk lines together
        chunks = []
        for w in range(0, len(sentences), chunk_size):
            chunks.append(' '.join(sentences[w:w+chunk_size]))
        
        # Save to lists outside the loop
        labels.extend(repeat(i, len(chunks))) # repeat season labels corresponding to the number of chunks
        text.append(chunks)                   # append chunks
    
    # From list of lists --> to list
    text = list(itertools.chain(*text))
    
    return(labels, text)


# Balance data 
def balance(dataframe, colname, n=500):
    """
    Function to balance data.
    dataframe = df to be balanced
    colname = column with labels the data should be balanced according to
    n = row per label
    
    The function is made by Ross a only slighty modified:
    https://github.com/CDS-AU-DK/cds-language/blob/main/utils/classifier_utils.py
    """

    # Use pandas select a random bunch of examples from each label
    out = (dataframe.groupby(colname, as_index=False)
            .apply(lambda x: x.sample(n=n))
            .reset_index(drop=True))
    
    return out


# balance chunks
def balance_chunks(labels, text):
    """
    Function using the already defined balance() function to balance the text chunks. 
    Return two lists containing labels and text
    """
    df = pd.DataFrame()
    df["season"] = labels
    df["line_chunks"] = text

    # Get the minimum amount of line chunks for a season - balance accordingly
    min_count = df["season"].value_counts().min()
    
    # Use predefined balance() function to balance amount of entrances across seasons
    balanced = balance(df, "season", min_count)

    # Extract information from the df
    line_chunks = balanced["line_chunks"]
    season = balanced["season"]
    
    return(season, line_chunks)


#Function to save classification report and print to terminal
def Report(y_test, y_pred, filename):
    # Get report
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    
    # Print it in the terminal
    print(classifier_metrics)
    
    # Create report object, turn it into a df, and save it as a csv file.
    report = metrics.classification_report(y_test, y_pred, output_dict = True)
    report = pd.DataFrame(report).transpose()
    report.to_csv(filename)
    print("The classification report has successfully been saved in the folder 'out'.")


#________MAIN FUNCTION________#

def main(args):
    
    # Import chunk_size specified in the commandline
    chunk_size = args["chunk_size"]
    test_size = args["test_size"]
    
    # Read the data as 'DATA' from the data folder
    filename = os.path.join("data", "Game_of_Thrones_Script.csv")
    DATA = pd.read_csv(filename, index_col=0)
    
    # Chunk up the lines to enable appropriate vectorisation and thus classification
    labels, text = chunk_me(DATA, chunk_size)
    
    # Balance the chunks so all seasons are represented with the same amount of chunks
    season, line_chunks = balance_chunks(labels, text)
    
    # Split the data in train and test.
    X_train, X_test, y_train, y_test = train_test_split(line_chunks,           # sentences for the model
                                                        season,                # classification labels (seasons)
                                                        test_size=test_size,   # Size specified in the commandline
                                                        random_state=42)       # random state for reproducibility
    
    # Create vector object
    vectorizer = TfidfVectorizer(ngram_range = (1,2),     # unigrams and bigrams (1 word and 2 word units)
                                 lowercase =  True,       # should not be sensitive to capital letters
                                 max_df = 0.95,           # remove very common words
                                 min_df = 0.05,           # remove very rare words
                                 max_features = 500)      # keep only top features
    
    # I use this vectorizer to turn all sentences in the data into vectors of numbers 
    # Both for train and test data
    X_train_feats = vectorizer.fit_transform(X_train)
    X_test_feats = vectorizer.transform(X_test)
    
    # List of feature names. 
    feature_names = vectorizer.get_feature_names()
    
    # Classifying and predicting
    classifier = LogisticRegression(random_state=42).fit(X_train_feats, y_train)
    y_pred = classifier.predict(X_test_feats)

    # Print classification report and save to out folder
    Report(y_test, y_pred, "out/LRM_ClassificationReport.csv")

    # Save heatmap confusion matrix
    y_test = pd.Series(y_test)
    plot_cm(y_test, y_pred, "out/LRM_ConfusionMatrix.png", normalized=True)


# Execute main() function
if __name__=="__main__":
    
    # Argument parser (test size and chunk size can be specified in the commandline)
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--chunk_size", type = int, default = 20,
                    help = "Specify the amount of lines that should be chunked together. The default is 20.")
    ap.add_argument("-t", "--test_size", type = float, default = 0.25,
                    help = "Specify the test size of the data. The default is 0.25.")
    
    # Parse arguments. args is now an object containing all arguments added through the terminal. 
    argument_parser = vars(ap.parse_args())
    
    # run main() function
    main(argument_parser)



