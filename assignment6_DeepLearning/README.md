# Assignment 6 - Deep Learning

## Description of the assignment
__Text classification using Deep Learning__

Winter is... hopefully over.

In class this week, we've seen how deep learning models like CNNs can be used for text classification purposes. For your assignment this week, I want you to see how successfully you can use these kind of models to classify a specific kind of cultural data - scripts from the TV series Game of Thrones.

You can find the data here: https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons

In particular, I want you to see how accurately you can model the relationship between each season and the lines spoken. That is to say - can you predict which season a line comes from? Or to phrase that another way, is dialogue a good predictor of season?

Start by making a baseline using a 'classical' ML solution such as CountVectorization + LogisticRegression and use this as a means of evaluating how well your model performs. Then you should try to come up with a solution which uses a DL model, such as the CNNs we went over in class.


## Repository structure and files

This repository has the following directory structure:

| Column | Description|
|--------|:-----------|
```out``` | Contains the outputs from running the scripts.
```LogisticRegressionModel.py```| The first script containing my baseline model which is a 'classical' ML solution such with logistic regression.
```DeepLearningModel.py```| The second script containing my deep learning solution.
```./create_lang_venv.sh``` | A bash script which automatically generates a new virtual environment 'DeepLearning_env', and install all the packages contained within 'requirements.txt'
```requirements.txt``` | A list of packages along with the versions that are required for the scripts to run.
```README.md``` | This very readme file


## Run the script
Setup the virtual environment and activate it
```bash
git clone https://github.com/askesvane/LanguageAnalytics.git
cd assignment6_DeepLearning
bash ./create_lang_venv.sh
source ./DeepLearning_env/bin/activate
```
In order to run the script 'DeepLearningModel.py', one needs to download an unzip the pretrained weights of the 'glove' library. Please execute from the terminal while being in the folder 'assignment6_DeepLearning' (the zipfile will take 3-4 minutes to download):

```bash
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip -q glove.6B.zip
```

Run the scripts. You can specify the number of lines that should be chunked together.
```bash
python LogisticRegressionModel.py --chunk_size
python DeepLearningnModel.py --chunk_size
```
The classification report and plots can be found in the folder called 'out'.

## Results

| Script | Result|
|--------|:-----------|
```LogisticRegressionModel.py``` | The accuracy is 0.28. Please find the classification report in the folder 'out'.
```DeepLearningModel.py```| The training accuracy is 1.0000 indicating a vast overfitting. The test accuracy is 0.2687. 
