# Assignment 3 - Sentiment Analysis

## Description of the assignment

Download the following CSV file from Kaggle: https://www.kaggle.com/therohk/million-headlines

This is a dataset of over a million headlines taken from the Australian news source ABC (Start Date: 2003-02-19 ; End Date: 2020-12-31).

- Calculate the sentiment score for every headline in the data. You can do this using the spaCyTextBlob approach that we covered in class or any other dictionary-based approach in Python.
- Create and save a plot of sentiment over time with a 1-week rolling average
- Create and save a plot of sentiment over time with a 1-month rolling average
- Make sure that you have clear values on the x-axis and that you include the following: a plot title; labels for the x and y axes; and a legend for the plot
- Write a short summary (no more than a paragraph) describing what the two plots show. You should mention the following points: 1) What (if any) are the general trends? 2) What (if any) inferences might you draw from them?

## Feedback instructions

- Clone the whole repository to a chosen location on your computer
- Through the terminal, navigate to the folder: ../LanguageAnalytics/assignment3_SA

### Set up virtual environment

- Execute in the terminal: ```bash create_lang_venv.sh```. This will create a virtual environment called ```sentiment_environment``` which will install everything in the ```requirements.txt``` file.
- Activate the virtual environment by executing ```sentiment_environment/bin/activate```. You should now see that the environment is activated at your command line.
- Run the .py script by executing ```python sentiment.py``` (one might have to write 'python3' instead of python). Note that the folder contains a .ipynb file as well and this script should not be run.
- The .py script will run on a subset of 10.000 headlines which will take approx. 1 min. This can be adjusted at line 104 in the script. After the script has run, two plots can be found in the folder 'plots' within 'assignment3_SA'.
- In case you have issues setting up the virtual environemnt locally on your computer and thus cannot run the script, I have left two plots in the folder based on the 10.000 headline subset.

### Summary describing the two plots 

The two plots are showing the average sentiment scores of newspaper headlines in the period 2003-2021. One plot shows the sentiment scores averages on a 1 week rolling average and the other on a 1 month rolling average. An increasing amount of days grouped together simplifies the plot enhancing interpretability. Thus, the 1 week rolling average plot contains substantially more flucturations compared to the 1 month rolling average plot.
- From the 1 week rolling average plot, it is extremely difficult to draw conclusions on general trends as any potential trends seem to drown in 'noisey' flucturations.
- This is to a great extent the same case in the 1 month rolling average plot. However, there seems to be periods in time where the headlines are generally more positive and periods where they are generally more negative. A general increase/decrease across years do not seem to be apparent. 










