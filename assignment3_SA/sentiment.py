
## Assignment on sentiment scores from headlines in the period 2003-2021


#_______________# Import packages #_______________# 


# import packages
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


# Function to change the date column into an actual time object
def date_creator(DF):
    
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

# Function to calculate sentiment scores for every headline in a df and append in a new column
def sentiment_calculator(DF):
    
    # Create empty df for the sentiment scores
    scores = []

    # Loop over every headline
    for doc in nlp.pipe(DF["headline_text"]):
    
        #for each headline, append the sentiment score to the 'scores' list
        scores.append(doc._.sentiment.polarity)

    # append the list as a column in the original dataframe
    DF["sentiment_score"] = scores
    
    return(DF)

# Function to create plots
def plot_creator(DF, days, title):
    
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

    figure_name = 'plots/{}_days_rolling.png'.format(days)
    plt.savefig(figure_name)
    
    plt.clf()


def main():
    
    # Create a data path
    in_file = os.path.join("data", "abcnews-date-text.csv")
    # read in as pandas df
    data = pd.read_csv(in_file)

    data = data.sample(1000)
    
    # Create date object in df
    data = date_creator(data)
    
    # Get sentiment scores
    data = sentiment_calculator(data)
    
    # Plots
    plot_creator(data, 7, "Mean sentiment score over time with a 1 week rolling average")
    plot_creator(data, 30, "Mean sentiment score over time with a 1 month rolling average")
    
    

#_______________# The end #_______________#
    
if __name__=="__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    



