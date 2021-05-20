#!/usr/bin/env python

# _________ Packages _________#

# system tools
import os
import sys
import argparse
sys.path.append(os.path.join(".."))

# pandas, numpy, gensim
import pandas as pd
import numpy as np

# Machine learning stuff
from sklearn.model_selection import train_test_split

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

### _________ THE SCRIPT _________###


### FUNCTION

# Chunk function
def chunk_me(DATA, chunk_size, predict, topCategories):
    """
    Function that will take the data and chunk size as well as the parameter 'predict' specified in the terminal 
    and a list of the possible outcomes. It returns two lists, one with labels and one with corresponding text 
    chunks given the chunk size defined
    
    This function is made by Aske for a previous assignment and has only been slightly modified:
    https://github.com/askesvane/LanguageAnalytics/tree/main/assignment6_DeepLearning
    """

    # Create empty list to save text chunks and corresponding labels
    text = []
    labels = []

    # Loop over every category and chunk lines together + save category label.
    for i in topCategories:
        
        # Get data from category
        category = DATA[DATA[predict] == i]
        
        # Get lines
        descriptions = category["description"]
        
        # Chunk lines together
        chunks = []
        for w in range(0, len(descriptions), chunk_size):   
            chunks.append(' '.join(descriptions[w:w+chunk_size]))
        
        # Save to lists outside the loop (repeat the name of the grape corresponding to the number of chunks)
        labels.extend(repeat(i, len(chunks)))
        text.append(chunks)
    
    # From list of lists --> to list
    text = list(itertools.chain(*text))
    
    return(labels, text)


# Balance data 
def balance(dataframe, colname, n=500):
    """
    The function is made by Ross and only slighty modified:
    https://github.com/CDS-AU-DK/cds-language/blob/main/utils/classifier_utils.py
    
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


# Function to create embedding matrix
def create_embedding_matrix(filepath, word_index, embedding_dim):
    """ 
    A helper function to read in saved GloVe embeddings and create an embedding matrix.
    We use the embedding matrix as weights (pretrained word embeddings rather 
    than just learning weights from our own data)
    
    filepath: path to GloVe embedding
    word_index: indices from keras Tokenizer
    embedding_dim: dimensions of keras embedding layer
    
    The Function has been made by Ross:
    https://github.com/CDS-AU-DK/cds-language/blob/main/notebooks/session10.ipynb
    """
    vocab_size = len(word_index) + 1  # Adding 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


# Padding function
def padding(X_train_toks, X_test_toks, maxlen):
    """
    Function that takes the tokenized train and test data as well as the max length. 
    Both the train and the test data is padded to correspond to the max chunk length. 
    The function returns the padded train and test data.
    """
    
    # pad training data to maxlen
    X_train_pad = pad_sequences(X_train_toks, 
                                padding='post', # sequences can be padded "pre" or "post"
                                maxlen=maxlen)  # maximum length
    # pad testing data to maxlen
    X_test_pad = pad_sequences(X_test_toks, 
                               padding='post', 
                               maxlen=maxlen)

    return(X_train_pad, X_test_pad)

# Plot function
def plot_history(H, epochs, filename):
    """
    Utility function for plotting model history using matplotlib
    H: model history 
    epochs: number of epochs for which the model was trained
    
    The function has been made by Ross and only slightly motified to save the output:
    https://github.com/CDS-AU-DK/cds-visual/blob/main/notebooks/session9.ipynb
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
    plt.savefig(f"out/{filename}")

# report function
def save_report(accuracy_train, accuracy_test, loss_train, loss_test, filename):
    """
    Function that takes the loss and accuracy of the train and test, gathers the information in a df,
    and saves it as a csv in the 'out' folder.
    """
    # Gather accuracy and loss in two lists
    accuracy = [accuracy_train, accuracy_test]
    loss = [loss_train, loss_test]
    # create df
    df = pd.DataFrame()
    df["dataset"] = ["train", "test"]
    df["accuracy"] = accuracy
    df["loss"] = loss
    # save df
    df.to_csv(f"out/{filename}", index=False)

# Function to remove words
def word_remove(text, wordlist):
    """
    This function takes a list of text strings and a list of words. It returns the list of text strings 
    with the words appearing in the wordlist removed.
    """
    # Create new emoty list for the cleaned text
    clean_text = []
    # Seperate ALL words from each other and write them with lower case
    wordlist = ' '.join(wordlist).lower().split()
    
    # Loop over every chunk and remove the words from the wordlist
    for i in range(len(text)):
        chunk = text[i] # take chunk
        chunkwords = chunk.split() # split into single words
        resultwords  = [word for word in chunkwords if word.lower() not in wordlist] # remove words from wordlist
        chunkclean = ' '.join(resultwords) # gather all words into a string again 
        clean_text.append(chunkclean) # append the chunk to the list outside the loop

    return(clean_text)



### THE MAIN FUNCTION

def main(args):
    
    # Import parameters specified in the terminal
    chunk_size = args["chunk_size"]
    n = args["number"]
    test_size = args["test_size"]
    embedding_dim = args["embedding_dim"]
    epochs = args["epochs"]
    predict = args["predict"]
    
    # Comment to terminal
    print(f"The deep learning model will be trained to predict '{predict}' from wine descriptions only.")
    
    # The column with grapes in the data is called "variety"
    if (predict == "grape"):
        predict = "variety"
        pass
    elif (predict == "province"):
        pass
    
    # Comment to terminal
    print("Importing the data...")
    
    # Read in the data as 'DATA' from the data folder
    filename = os.path.join("data", "winemag-data-130k-v2.csv")
    DATA = pd.read_csv(filename, index_col=0)
    
    # Comment to terminal
    print("The data is being preprocessed before being fed to the model...")
    
    # get top most frequent categories
    topCategories = DATA[predict].value_counts()[:n].index.tolist()
    
    # Only keep top categories in the data
    DATA_top = DATA.loc[DATA[predict].isin(topCategories)]
    
    # Chunk together descriptions according to the chunk size specified in the terminal
    labels, text = chunk_me(DATA_top, chunk_size, predict, topCategories)
    
    # Remove category words from the descriptions.
    # The label to be predicted should not appear in the description.
    text = word_remove(text, topCategories)
    
    # Create df to hold the data (as the function balance() takes a df as the input)
    df = pd.DataFrame()
    df["category"] = labels
    df["line_chunks"] = text
    
    # Get the minimum amount of line chunks for a category - and balance accordingly
    min_count = df["category"].value_counts().min()
        
    # Use predefined balance() function to balance amount of entrances across descriptions
    balanced = balance(df, "category", min_count)
    
    # Extract information from the df
    line_chunks = balanced["line_chunks"]
    category = balanced["category"]
    
    # Create a new column 'totalwords' in balanced with the word count of the chunk.
    balanced['totalwords'] = balanced['line_chunks'].str.split().str.len()
    
    # Get the max length of a chunk based on the new word column
    maxlen = int(balanced[balanced['totalwords'] == balanced['totalwords'].max()]['totalwords'])
    
    # Split the data in train and test.
    X_train, X_test, y_train, y_test = train_test_split(line_chunks,           # text for the model
                                                        category,                
                                                        test_size=test_size,   
                                                        random_state=42)       # random state for reproducibility
    
    # Change the split data into numpy arrays
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    # Binarize the labels
    lb = LabelBinarizer() 
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    
    # initialize tokenizer
    tokenizer = Tokenizer(num_words = 5000)
    
    # Tokenize train and test data according to the tokenizer
    # fit to training data - fit to the raw text data. vocabolary of 10000 words now fit onto our training data
    tokenizer.fit_on_texts(X_train)
    
    # tokenized training and test data
    X_train_toks = tokenizer.texts_to_sequences(X_train)
    X_test_toks = tokenizer.texts_to_sequences(X_test)
    
    # overall vocabulary size
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    
    # fit the tokenizer on to the raw text data.
    tokenizer.fit_on_texts(X_train)
    
    # tokenized training and test data
    X_train_toks = tokenizer.texts_to_sequences(X_train)
    X_test_toks = tokenizer.texts_to_sequences(X_test)
    
    # overall vocabulary size
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    # Padding
    X_train_pad, X_test_pad = padding(X_train_toks, X_test_toks, maxlen)

    # Comment to terminal
    print("Pretrained weights from 'GloVe' are being imported...")
    
    #create embedding matrix with the words appearing in the data. 
    embedding_matrix = create_embedding_matrix('glove.6B.50d.txt',
                                                   tokenizer.word_index, 
                                                   embedding_dim)
    
    # Create regularizer 
    l2 = L2(0.0001)
    
    # Defining and compiling model
    model = Sequential()
        
    # Embedding -> CONV+ReLU -> MaxPool -> FC+ReLU -> Out
    model.add(Embedding(vocab_size,                  # vocab size from Tokenizer()
                        embedding_dim,               # embedding input layer size
                        weights=[embedding_matrix],  # pretrained embeddings
                        input_length=maxlen,         # maxlen of padded doc
                        trainable = True))           # trainable embeddings
    
    model.add(Conv1D(128, 5, # added at the convolutional layer
                    activation='relu',
                    kernel_regularizer=l2))          # L2 regularization 
    model.add(GlobalMaxPool1D())
    model.add(Dense((n*4), activation='relu', kernel_regularizer=l2))
    model.add(Dense(n, activation='softmax'))
        
    # compile
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    
    # Comment to terminal
    print("The data is fit to the model...")
          
    # Fit the model
    history = model.fit(X_train_pad, y_train,
                    epochs=epochs,
                    verbose=False,
                    validation_data=(X_test_pad, y_test),
                    batch_size=10)
    
    # evaluate and print accuracy to terminal
    loss_train, accuracy_train = model.evaluate(X_train_pad, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy_train))
    loss_test, accuracy_test = model.evaluate(X_test_pad, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy_test))
    
    # Create filenames depending on whether grapes or provinces are being predicted.
    if (predict == "variety"):
        img_results = "grape_results.csv"
        img_plot = "grape_TraininglossAccuracy.jpg"
    elif (predict == "province"):
        img_results = "province_results.csv"
        img_plot = "province_TraininglossAccuracy.jpg"
    
    # Save plot of accuracy and loss, model illustration as well as accuracy and loss scores in the folder 'out'
    save_report(accuracy_train, accuracy_test, loss_train, loss_test, img_results)
    plot_history(history, epochs = epochs, filename = img_plot)
    plot_model(model, to_file='out/Model.jpg', show_shapes=True, show_layer_names=True)
          
    print(f"Accuracy and loss scores have been saved in the folder 'out' as '{img_results}' along with a plot '{img_plot}'. An illustration of the model 'Model.jpg' has as well been saved in the folder.")
    

# RUN THE MAIN FUNCTION
if __name__=="__main__":
    
    # Argument parser
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-n", "--number", type = int, default = 10,
                    help = "Specify the amount of categories that should be included in the classification task. Default = 10")
    ap.add_argument("-c", "--chunk_size", type = int, default = 10,
                    help = "Specify the amount of lines that should be chunked together. Default = 10")
    ap.add_argument("-t", "--test_size", type = float, default = 0.25,
                    help = "Specify the test size. Default = 0.25")
    ap.add_argument("-d", "--embedding_dim", type = int, default = 50,
                    help = "Specify the number of embedding dimensions. Default = 50")
    ap.add_argument("-e", "--epochs", type = int, default = 5,
                    help = "Specify the number of epochs. Default = 5")
    ap.add_argument("-p", "--predict", type = str, required=True,
                    help = "Specify what the model should predict from the wine descriptions. The parameter can either be specified as 'grape' or 'province'.")
    
    # Parse arguments. args is now an object containing all arguments added through the terminal. 
    argument_parser = vars(ap.parse_args())
    
    # run main() function
    main(argument_parser)

