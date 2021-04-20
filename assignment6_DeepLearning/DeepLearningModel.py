#!/usr/bin/env python

# _________ Install packages _________#

# system tools
import os
import sys
import argparse
sys.path.append(os.path.join(".."))

# pandas, numpy, gensim
import pandas as pd
import numpy as np
import gensim.downloader

# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, 
                                     Flatten, GlobalMaxPool1D, Conv1D)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import L2

# sklearn tools
from sklearn.preprocessing import LabelBinarizer

# additional packages
import matplotlib.pyplot as plt
from itertools import repeat
import itertools


# _________ The script _________#


### FUNCTIONS

# Function to create embedding matrix
def create_embedding_matrix(filepath, word_index, embedding_dim):
    """ 
    A helper function to read in saved GloVe embeddings and create an embedding matrix
    
    filepath: path to GloVe embedding
    word_index: indices from keras Tokenizer
    embedding_dim: dimensions of keras embedding layer
    """
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

# Balance data 
def balance(dataframe, colname, n=500):
    """
    Function to balance data.
    dataframe = df to be balanced
    colname = column with labels the data should be balanced according to
    n = row per label
    """
    # Use pandas select a random bunch of examples from each label
    out = (dataframe.groupby(colname, as_index=False)
            .apply(lambda x: x.sample(n=n))
            .reset_index(drop=True))
    
    return out

# balance chunks
"""
Function using the already defined balance() function to balance the text chunks. 
Return two lists containing labels and text"""
def balance_chunks(labels, text):
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
    
    return(season, line_chunks, balanced)


# Chunk function
"""
Function that will take the data and return to lists, 
one with labels and one with corresponding text chunks given the chunk size defined"""
def chunk_me(DATA, chunk_size):
    
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
        labels.extend(repeat(i, len(chunks)))
        text.append(chunks)
    
    # From list of lists --> to list
    text = list(itertools.chain(*text))
    
    return(labels, text)


# WORD EMBEDDINGS
def word_embedding(X_train, X_test, tokenizer):
    """
    Function that takes training and test data and returns them as tokenized"""
    
    # fit to training data - fit to the raw text data. vocabolary of 10000 words now fit onto our training data
    tokenizer.fit_on_texts(X_train)
    
    # tokenized training and test data
    X_train_toks = tokenizer.texts_to_sequences(X_train)
    X_test_toks = tokenizer.texts_to_sequences(X_test)
    
    # overall vocabulary size
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    return(X_train_toks, X_test_toks, vocab_size)

### Padding
def pad_me(X_train_toks, X_test_toks, balanced):
    
    balanced['totalwords'] = balanced['line_chunks'].str.split().str.len()
    maxlen = int(balanced[balanced['totalwords'] == balanced['totalwords'].max()]['totalwords'])

    # pad training data to maxlen
    X_train_pad = pad_sequences(X_train_toks, 
                                padding='post', # sequences can be padded "pre" or "post"
                                maxlen=maxlen)  # maximum length
    # pad testing data to maxlen
    X_test_pad = pad_sequences(X_test_toks, 
                               padding='post', 
                               maxlen=maxlen)

    return(X_train_pad, X_test_pad, maxlen)


def model_creator(vocab_size, embedding_dim, embedding_matrix, maxlen, l2):
    model = Sequential()
    
    # Embedding -> CONV+ReLU -> MaxPool -> FC+ReLU -> Out
    model.add(Embedding(vocab_size,                  # vocab size from Tokenizer()
                        embedding_dim,               # embedding input layer size
                        weights=[embedding_matrix],  # pretrained embeddings
                        input_length=maxlen,         # maxlen of padded doc
                        trainable=True))             # trainable embeddings
    
    model.add(Conv1D(128, 5, # added at teh convolutional layer
                    activation='relu',
                    kernel_regularizer=l2))          # L2 regularization 
    model.add(GlobalMaxPool1D())
    model.add(Dense(32, activation='relu', kernel_regularizer=l2))
    model.add(Dense(8, activation='softmax'))
    
    # compile
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    
    return(model)

def plot_history(H, epochs):
    """
    Utility function for plotting model history using matplotlib
    
    H: model history 
    epochs: number of epochs for which the model was trained
    """
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("out/DLM_TraininglossAccuracy.png")

### MAIN FUNCTION

def main(args):
    
    # Import chunk_size specified in the commanline
    chunk_size = args["chunk_size"]
    
    # Read the data as 'DATA' from the data folder
    filename = os.path.join("data", "Game_of_Thrones_Script.csv")
    DATA = pd.read_csv(filename, index_col=0)
    
    # Chunk up the lines to enable appropriate vectorisation and thus classification
    labels, text = chunk_me(DATA, chunk_size)
    
    # Balance the chunks so all seasons are represented with the same amount of chunks
    season, line_chunks, balanced = balance_chunks(labels, text)
    
    # Split the data in train and test.
    X_train, X_test, y_train, y_test = train_test_split(line_chunks,     # sentences for the model
                                                        season,          # classification labels (seasons)
                                                        test_size=0.25,   # create an 70/30 split
                                                        random_state=42) # random state for reproducibility

    # Change the split data into numpy arrays
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    # Binarize the labels
    lb = LabelBinarizer() 
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    
    # Word embedding - Change the strings of texts into numerical representations taking context into account
    tokenizer = Tokenizer(num_words = 10000) # initialize tokenizer
    X_train_toks, X_test_toks, vocab_size = word_embedding(X_train, X_test, tokenizer)
    
    # Padding
    X_train_pad, X_test_pad, maxlen = pad_me(X_train_toks, X_test_toks, balanced)
    
    # Word embedding matrix
    """create embedding matrix with the words that appear in the data. 
    We use the embedding matrix as weights (pretrained word embeddings rather 
    than just learning weights from our own data)"""
    embedding_dim = 50
    embedding_matrix = create_embedding_matrix('glove.6B.50d.txt',
                                               tokenizer.word_index, 
                                               embedding_dim)
    
    # Create regularizer 
    l2 = L2(0.0001)
    
    
    ### Create and compile the model
    model = model_creator(vocab_size, embedding_dim, embedding_matrix, maxlen, l2)

    # Fit the model
    history = model.fit(X_train_pad, y_train,
                    epochs=20,
                    verbose=False,
                    validation_data=(X_test_pad, y_test),
                    batch_size=10)

    # evaluate and print accuracy to terminal
    loss, accuracy = model.evaluate(X_train_pad, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    # Save plot of accuracy and loss in the out folder
    plot_history(history, epochs = 20)


    
# Execute main() function
if __name__=="__main__":
    
    # Argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--chunk_size", type = int, default = 5,
                    help = "Specify the amount of lines that should be chunked together. Default = 5")
    
    # Parse arguments. args is now an object containing all arguments added through the terminal. 
    argument_parser = vars(ap.parse_args())
    
    # run main() function
    main(argument_parser)
