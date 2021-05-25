#!/usr/bin/env python

#_______________# Import packages #_______________# 

# import packages
import argparse
import spacy
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from spacytextblob.spacytextblob import SpacyTextBlob
nlp=spacy.load("en_core_web_sm")

# add spacy text blob as a new component in the nlp pipeline
spacy_text_blob = SpacyTextBlob()
nlp.add_pipe(spacy_text_blob)


#_______________# The script #_______________#

### FUNCTIONS

# date_function: 
def date_creator(DF):
    """
    Function that changes the date column into an actual time object.
    The function takes the dataframe as the only parameter.
    """
    # Creating empty list for the transformation of dates
    all_dates = []

    # loop over every row in the df
    for index, row in DF.iterrows():
    
        # extract the time as a string of numbers
        date = str(row['publish_date'])

        # By indexing, extract year, month, and day
        time_df = pd.DataFrame({
            'year': [int(date[0:4])],
            'month': [int(date[4:6])],
            'day': [int(date[6:8])]})
        
        # Create as a time object
        time_df = pd.to_datetime(time_df)
    
        # Append to the list 'all_dates'
        all_dates.append(time_df[0])

    # Overwrite the original publish_date column with the content of the list
    DF["publish_date"] = all_dates
    
    return(DF)


# Sentiment calculater function:
def sentiment_calculator(DF):
    """
    Function to calculate sentiment scores for every headline in a df and append in a new column.
    The function takes the dataframe as the only parameter.
    """
    # Create empty df for the sentiment scores
    scores = []

    # Loop over every headline
    for doc in nlp.pipe(DF["headline_text"]):
    
        #for each headline, append the sentiment score to the 'scores' list
        scores.append(doc._.sentiment.polarity)

    # append the list as a column in the original dataframe
    DF["sentiment_score"] = scores
    
    return(DF)

# Function plot creator:
def plot_creator(DF, days, title):
    """
    Function that creates the plots with rolling time averages.
    It takes the the parameters:
    - a dataframe
    - Number of days it should average over in the plot
    - A plot title.
    """
    # Create a dataframe grouped by dates (only one row per date with a mean sentiment score)
    grouped_df = DF.groupby("publish_date").mean("sentiment_scores")

    # Plot it with a 7 day rolling average
    plt.plot(grouped_df.rolling(days).mean())

    #add title 
    plt.title(title)

    #add xlabel 
    plt.xlabel("Time")

    #add ylabel 
    plt.ylabel("Mean sentiment score")

    # Save the figure
    figure_name = 'plots/{}_days_rolling.png'.format(days)
    plt.savefig(figure_name)
    plt.clf()
    
    return

### MAIN FUNCTION
def main(args):
    
    # Message to terminal
    print("Importing and pre-processing the data...")
    
    # Create a data path
    in_file = os.path.join("data", "abcnews-date-text.csv")
    
    # read in as pandas df
    data = pd.read_csv(in_file)

    # Subset the data given what has been specified in the command line.
    subset = args["subset"]
    print(f"The visualisations will be created on a sample of {subset} headlines.")
    data = data.sample(subset)
    
    # Create date object in df
    data = date_creator(data)
    
    # Message to terminal
    print("Calculating sentiment scores...")
    
    # Get sentiment scores
    data = sentiment_calculator(data)
    
    # Message to terminal
    print("Creating plots...")
    
    # Plots
    plot_creator(data, 7, "Mean sentiment score over time with a 1 week rolling average")
    plot_creator(data, 30, "Mean sentiment score over time with a 1 month rolling average")
    
    # Message to terminal
    print("The plots have successfully been saved in the folder 'plots' as '7_days_rolling.png' and '30_days_rolling.png'.")
    
### RUN MAIN
    
if __name__=="__main__":
    
    # Argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--subset", type = int, default = 10000,
                    help = "Specify the size of the subset. The default is 10,000.")

    # Parse arguments. args is now an object containing all arguments added through the terminal. 
    argument_parser = vars(ap.parse_args())
    
    # run main() function
    main(argument_parser)