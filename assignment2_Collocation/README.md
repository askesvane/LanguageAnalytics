# Assignment 2 - Collocation

__Contribution__<br>
The code of this assignment was created by me.

## Description of the assignment

__String processing with Python__

Using a text corpus found on the cds-language GitHub repo or a corpus of your own found on a site such as Kaggle, write a Python script which calculates collocates for a specific keyword.

- The script should take a directory of text files, a keyword, and a window size (number of words) as input parameters, and an output file called out/{filename}.csv.
- Find out how often each word collocates with the target across the corpus
- Use this to calculate mutual information between the target word and all collocates across the corpus
- Save result as a single file consisting of three columns: collocate, raw_frequency, MI
- BONUS CHALLENGE: Use argparse to take inputs from the command line as parameters

## Methods

As the task is to create an output file containing every collocate, their raw frequencies as well as MI, I need to calculate the MI. In order to do so, I first have to calculate the observed co-occurance frequency and the expected co-occurance frequency of each word co-occuring with the keyword. I follow the documentation [here](http://collocations.de/AM/index.html). These observed and expected frequencies are then employed to calculate the MI; the MI is calculated as the log of the observed frequency divided by the expected frequency.

## Evaluation

The command line tool successfully returns a csv-file in the folder 'out' with the information 'collocate', 'raw_frequency', and 'MI'. The rows are sorted after frequency where the words with lowest co-occurance frequency come first and highest co-occurance frequency come last.

__Minor considerations__<br>
The collocates are defined as appearing within a specified 'window' around the keyword measured in characters. Thus, all the words within the window are to the same degree considered as co-occuring with the keyword. In terms of meaning, there is however a difference between words appearing right next to each other and words appearing with several words between them. The window size is arbitrarily defined (the default is 30 characters on each side) and can be specified to be of any size in the command line. This could in turn alter the content of the output file dramatically as well as potential conclusions drawn from it.

The output file contains a lot of stopwords such as 'the', 'a', 'of', etc. from which it is hard to say anything about the keyword. To extract meaningful information from the output, I could have excluded these according to a stopword dictionary.

## Repository structure and files
This repository has the following directory structure:

| Column | Description|
|--------|:-----------|
```data```| A folder containing the data. The folder 'Stevenson' containing three texts of Robert Louis Stevenson has been provided. Any folder other folder could be added under 'data' and specified in the command line.
```out```| A folder containing the output csv-file from executing the script.
```collocation.py```| The script to be executed from the command line.
```create_lang_venv.sh```| A bash script to set up the virtual environment 'collocation_env'.
```README.md```| This file.
```requirements.txt```| All packages required to run the script. They will automatically be installed in the virtual environment when running the bash script.


## Usage (reproducing the results)

### Virtual environment
In order to run the script, one is required to set up the virtual environment with all necessary packages installed. Please clone the repo, navigate to the folder for this assignment, run the bash script to set up the environment, and lastly activate it. The following code should be executed from the terminal:

```bash
git clone https://github.com/askesvane/LanguageAnalytics.git
cd assignment2_Collocation
bash ./create_lang_venv.sh
source ./collocation_env/bin/activate
```

### Execute the script 
Now, the script can be executed. From the command line, one is *required* to specify a keyword (--keyword) to which the script should find collocations. Additionally, one can specify the window size (--window_size) in characters (the desfault is 30) and a folder (--file_path) in 'data' with txt-files on which the script should perform the analysis (the default is 'Stevenson').

```bash
python sentiment.py --keyword [write a word here] --window_size 30 --file_path Stevenson
```
After running, the output csv-file can be found in the folder ```out```.