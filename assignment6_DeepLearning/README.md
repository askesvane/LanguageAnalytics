# Assignment 6 - Deep Learning

## Description of the assignment
__Text classification using Deep Learning__

Winter is... hopefully over.
The assignment is to see how successfully you can use CNN’s to classify a specific kind of cultural data - scripts from the TV series Game of Thrones.

The data can be found [here](https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons).
I want you to see how accurately you can model the relationship between each season and the lines spoken. That is to say - can you predict which season a line comes from? Or to phrase that another way, is dialogue a good predictor of season?

Start by making a baseline using a 'classical' ML solution such as CountVectorization + LogisticRegression and use this as a means of evaluating how well your model performs. Then you should try to come up with a solution which uses a DL model, such as the CNNs we went over in class.

## The logistic regression model

### Methods 
First, the data is being pre-processed. Lines are chunked together according to the chunk size specified in the commandline. The default is 40 lines per chunk. In addition, the data is being balanced so all seasons are represented with the same amount of chunks. 

The data is split into a train and a test dataset according to the proportion also specified in the commandline. Here, the default is a test size of 0.25. The text is vectorized using TfidfVectorizer() only keeping the top 500 features and subsequently fed into a simple logistic regression model (using the LogisticRegression() function).

### Results and evaluation
With the default parameters (a chunk size of 40 and a test size of 0.25), the weighted accuracy of the model is 33%. With 8 classification possibilities, this is above change indicating that the model has learned something from the lines in terms of predicting the respective season. However, the model performance is still very low and any prediction of season can be considered fairly unrealiable. 

In addition, adjusting the chunk size is a moderation which could entail various issues one should be aware of. On the one hand, if the chunks are too small the model will have problems learning any patterns from the lines that could predict the respective season. There would simply not be information enough. On the other hand, if the chunks are too long the number of chunks per season will decrease dramatically. The model will then only we trained on very few chunks per season.

## The deep learning model (CNN)

### Methods
The first pre-processing steps are identical to the pre-processing steps for the logistic regression model. The data is being chunked together according to the chunk size specified in the command line. The data is subsequently balanced by season and divided into a train and a test dataset.

With keras Tokenizer() the text strings are converted into numbers. I define the length of the corpus to be 10.000 regardless of the actual length. The text chunks are afterwards 'padded' so they are of equal length corresponding to the longest text chunk. 

The numerical representations are then converted into a dense embedded representation which facilitates that context can be taken into consideration. The library ```GloVe``` is used to enhance the model as every word in the corpus is mapped onto pretrained embeddings from GloVe. Thus, the model incorporates pretrained weights rather than only learning new weights from the data. The number of embedding dimensions is defined as 50 and the text now has a 2D dimensionality.

__The model__<br>
The data is first fed into an embedding layer with the pretrained GloVe weights of the embedding matrix. The embedding layer is followed by a convolution layer employing the 'ReLU' activation function and with L2 regularization of 0.0001. This will constrain the model to learn a more regular set of weights in order to prevent overfitting. Then follows a max pooling layer and two dense layers, the first with a 'ReLU' activation function and the second with a 'softmax' activation function with 8 potential outcomes (an illustration of the model can be found in the folder 'out' as 'DLM_Model.jpg').
The model is compiled using 'Adam' as the optimizer and 'categorical crossentropy' as the loss function parameter. 


### Results and evaluation
The training accuracy is 1.0 while the test accuracy is 0.24. The model is thus performing worse than the simple logistic regression model. 

Apart from being a very badly performing model in terms of predicting accuracy, the training accuracy of 1.0 indicates that the model is overfitting to the training data. The plot 'DLM_TraininglossAccuracy.png' also shows the overfitting. After just 2 epochs the training- and test accuracy graphs start to diverge with the test accuracy graph stagnating around 0.25 while the training accuracy graph keeps increasing until reaching 1.0.

The complexity of this deep learning classification model is quite high and modifying different parameters could potentially increase model performance while diminishing overfitting. With this kept in mind, the extremely low performance however indicates that the dialogue in 'Game of Thrones' is not a good predictor of season.

## Repository structure and files

This repository has the following directory structure:

| Column | Description|
|--------|:-----------|
```data``` | Folder containing the dialogue from GOT.
```out``` | Contains the outputs from running the scripts.
```DeepLearningModel.py```| The script containing my deep learning solution.
```LogisticRegressionModel.py```| The script containing my baseline model which is a 'classical' logistic regression ML solution.
```README.md``` | This very readme file.
```./create_lang_venv.sh``` | A bash script which automatically generates a new virtual environment 'DeepLearning_env', and install all the packages contained within 'requirements.txt'
```requirements.txt``` | A list of packages along with the versions that are required for the scripts to run.


## Usage (reproducing the results)

### Virtual environment
One is required to set up the virtual environment with all necessary packages installed specified in 'requirements.txt'. Please clone the repo, navigate to the folder for this assignment, run the bash script to set up the environment, and lastly activate it. The following code should be executed from the terminal:

```bash
git clone https://github.com/askesvane/LanguageAnalytics.git
cd assignment6_DeepLearning
bash ./create_lang_venv.sh
source ./DeepLearning_env/bin/activate
```
### Download the ```Glove``` library
To run the script 'DeepLearningModel.py', one needs to download an unzip the pretrained weights of the 'glove' library. Please execute the following from the terminal while being in the folder 'assignment6_DeepLearning' (the zipfile will take 3-4 minutes to download):

```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip -q glove.6B.zip
```
### Run the scripts
The scripts should be executed from the terminal. You can specify the number of lines that should be chunked together as well as the size of the test dataset. I have specified the default values.
```bash
python LogisticRegressionModel.py --chunk_size 20 --test_size 0.25
python DeepLearningnModel.py --chunk_size 20 --test_size 0.25
```
The classification reports and plots can be found in the folder called 'out'. In this folder, files starting with 'LRM_' relate to the logistic regression model and 'DLM_' relate to the deep learning model.