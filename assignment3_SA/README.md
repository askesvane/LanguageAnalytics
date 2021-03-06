# Assignment 3 - Sentiment Analysis

__Contribution__<br>
The code of this assignment was created by me.

## Description of the assignment

__Dictionary-based sentiment analysis with Python__

This assignment will be based on [this](https://www.kaggle.com/therohk/million-headlines) csv-file from kaggle. It is a dataset of over a million headlines taken from the Australian news source ABC (Start Date: 2003-02-19; End Date: 2020-12-31).

The assignment is to:
- Calculate the sentiment score for every headline in the data. You can do this using the spaCyTextBlob approach that we covered in class or any other dictionary-based approach in Python.
- Create and save a plot of sentiment over time with a 1-week rolling average
- Create and save a plot of sentiment over time with a 1-month rolling average
- Make sure that you have clear values on the x-axis and that you include the following: a plot title; labels for the x and y axes; and a legend for the plot
- Write a short summary (no more than a paragraph) describing what the two plots show. You should mention the following points: 1) What (if any) are the general trends? 2) What (if any) inferences might you draw from them?

## Methods 

In order to plot the mean sentiment of the headlines over time, the script employs an nlp pipeline to get the sentiment scores. For this task, I use spacy's pipeline "en_core_web_sm". In addition, I add the pipeline component "spaCyTextBlob" to employ the extension ```._.polarity``` when calculating the sentiment scores. The sentiment scores are averaged over both 7 days and 30 days and subsequently plotted with matplotlib.

## Results and evaluation

___Summary describing the two plots___

The two plots are showing the average sentiment scores of newspaper headlines in the period 2003-2021. One plot shows the sentiment scores averages on a 1 week rolling average and the other on a 1 month rolling average. An increasing amount of days grouped together simplifies the plot enhancing interpretability. Thus, the 1 week rolling average plot contains substantially more flucturations compared to the 1 month rolling average plot.
- From the 1 week rolling average plot, it is extremely difficult to draw conclusions on general trends as any potential trends seem to drown in noisy fluctuations.
- This is to a great extent the same case in the 1 month rolling average plot. However, there seems to be periods in time where the headlines are generally more positive and periods where they are generally more negative. A general increase/decrease across years do not seem to be apparent.

## Repository structure and files
This repository has the following directory structure:

| Column | Description|
|--------|:-----------|
```data```| A folder with the dataset from kaggle.
```plots```| A folder containing the two plots generated by the script.
```create_lang_venv.sh```| A bash script to set up the virtual environment 'sentiment_env'.
```README.md```| This file.
```requirements.txt```| All packages required to run the script. They will automatically be installed in the virtual environment when running the bash script.
```sentiment.py```| The script to be executed from the command line.

## Usage (reproducing the results)

### Virtual environment
In order to run the script, one is required to set up the virtual environment with all necessary packages installed. Please clone the repo, navigate to the folder for this assignment, run the bash script to set up the environment, and lastly activate it. The following code should be executed from the terminal:

```bash
git clone https://github.com/askesvane/LanguageAnalytics.git
cd assignment3_SA
bash ./create_lang_venv.sh
source ./sentiment_env/bin/activate
```

### Execute the script 
Now, the script can be executed. From the command line, one can specify the size of the subset sampled from the original data with over a million headlines. The default is 10,000.

```bash
python sentiment.py --subset 10000
```
After running, the two plots can be found in the folder ```plots```.










