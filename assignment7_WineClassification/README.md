# Assignment 7 - Wine Classification
__Self-assigned project__

## Description of the assignment

__Assignment__

The assignment is set to build a classification model which can predict the variety of grapes used to produce a certain wine as well as the province in which the wine is produced only based on its flavour description. 

__Contribution__

This assignment was written as a group project between Hanna Janina Matera (au603273) and Aske Svane Qvist (au613522), where: 

> “Both students contributed equally to every stage of this project from initial conception and implementation, through the production of the final output and structuring of the repository. (50/50%)”

__Background__

It is commonly known that the art of winemaking is an inseparable element of cultural identity for many nations around the world and that wine is one of the most frequently consumed alcoholic beverages. With the multitude of data on wine available online, it is possible to get a deeper insight into the field by building real-world tools helpful in the process of e.g., classifying the wine types or predicting the province of origin based on several wine characteristics. 
The creation of such classification tool is the main focus of the current assignment where the type of grapes used in the wine production process and the province of their origin are predicted based on the linguistic flavour descriptions given by professional sommeliers. 
A working classification model brings about a possibility of creating a real-life, user-friendly tool that could be used in commercial contexts, e.g., in a creation of an interactive tool where the most suitable wine type is suggested based on user’s preferred taste notes or where the wine growing province is estimated based on the descriptive information on taste. 

Thus, the two main questions asked in the current assignment are:

> Q1: Can the variety of grapes used to produce a given wine be predicted from the linguistic description of its flavor characteristics?

> Q2: Can the province of a given wine be predicted from the linguistic description of its flavor characteristics?


## Methods
Before constructing the model, the data is preprocessed. Since some of the wine descriptions are very short, the data is chunked together in order to facilitate better model estimation. The number of descriptions that should be chunked together, the chunk size, is a parameter that can be specified in the command line with a default size of 10. Subsequently, the data is balanced by either grape or province (depending on what is being predicted) so that all categories are represented with the same amount of chunks. From the command line, one can specify if either the grape or the province should be predicted.

Both the grape used to produce a given wine as well as the province in which the wine is produced are typically mentioned in the description of the wine. The grape or the province would most likely be predicted with a great accuracy solely because the predicted label is mentioned in the original text corpus already. To overcome this problem and to force the model to actually learn from the *description* of the taste experience, the categories to be predicted (either the grapes or the provinces) are removed from the text corpus. The prediction of the category labels will now be based on other features in the descriptions such as flavour notes, colour characteristics etc.

Upon necessary preprocessing the data is divided into a train and a test dataset. With keras Tokenizer() the text strings are then converted into numbers. The size of the corpus is defined to be 5.000 regardless of the actual size. The text chunks are afterwards 'padded' so they are of an equal length corresponding to the longest text chunk. Afterwards, the numerical representations are converted into a dense embedded representation enabling the textual context to be taken into consideration. The library GloVe (glove.6B.50d) is used to enhance the model as every word in the corpus is mapped onto pretrained embeddings from GloVe. Thus, the model incorporates pretrained weights rather than only learning new weights from the data. The number of embedding dimensions is by default set to 50 dimensions but can be defined from the command line.

__The model__

The preprocessed data is first fed into an embedding layer with the pretrained GloVe weights of the embedding matrix. The embedding layer is followed by a convolution layer employing the 'ReLU' activation function and with L2 regularization of 0.0001. This has been included to constrain the model to learn a more regular set of weights in order to prevent overfitting. Then follows a max pooling layer and two dense layers, the first with a 'ReLU' activation function and the second with a 'softmax' activation function with *n* potential outcomes (where n is the number of outcome categories which can be specified from the command line). The model is then compiled using 'Adam' as the optimizer and 'categorical crossentropy' as the loss function parameter.


## Results and evaluation

__Predicting the grape__ 

Running with default parameters, the training accuracy was predicted to be 99.9% while the test accuracy was 94.8%. Overall, the model is performing exceptionally well when predicting the specific grape variety.

When inspecting the training accuracy in the graph, the model seems to be slightly overfitting to the training data around the third epoch (see the plot ‘grape_TraininglossAccuracy.png’). 

__Predicting the province__

Running with default parameters, the training accuracy was predicted to be 99.9% while the test accuracy was 96%. Again, the model is performing extremely well when predicting the specific province of wine origin.

As was the case when predicting the grape, the model seems to be overfitting a bit. After about 2 epochs, the training and test accuracy as well as the training and test loss start to diverge (see the plot province_TraininglossAccuracy.jpg') indicating that the model is overfitting to the training data.

All results and figures with the outcome graphs can be found in the 'out' folder.

## Data

Data used in the current study was obtained from the online service for data scientist’s community, [Kaggle](https://www.kaggle.com/datasets). All data was retrieved as a single csv-file with 130,000 entries of different wines obtained through web-scraping of the popular wine-rating service, [WineEnthusiast](https://www.wineenthusiast.com/). Among other information, the data included a verbal description of each wine's flavor characteristics, the province of production as well as the grape, but also additional information such as the price per bottle, taste rating scores, and the name of the winery. The data can be viewed and downloaded [here](https://www.kaggle.com/zynicide/wine-reviews?select=winemag-data-130k-v2.csv).

## Repository structure and files
This repository has the following directory structure:

| Column | Description|
|--------|:-----------|
```data```| A folder with a csv-file containing information about wine and its descriptions.
```out``` | Contains the outputs with visuals from running the script.
```README.md``` | This readme file.
```create_lang_venv.sh``` | A bash script which automatically generates a new virtual environment 'Wine_env', and install all the packages contained within 'requirements.txt'
```requirements.txt``` | A list of packages along with the versions that are required.
```wine.py```| The script to be executed from the terminal.



## Usage (reproducing the results)

### Virtual environment
In order to run the script, one is required to set up the virtual environment with all necessary packages installed. Please clone the repo, navigate to the folder for this assignment, run the bash script to set up the environment, and lastly activate it. The following code should be executed from the terminal:

```bash
git clone https://github.com/askesvane/LanguageAnalytics.git
cd assignment7_WineClassification
bash ./create_lang_venv.sh
source ./Wine_env/bin/activate
```

### Download the ```GloVe``` library

In order to run the script 'wine.py', one needs to download an unzip the pretrained weights of the 'glove' library. Please execute the following from the terminal while being in the folder 'assignment7_WineClassification' (the zipfile will take 3-4 minutes to download):

```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip -q glove.6B.zip
```

### Execute the script 
Now, the script can be executed. The '--number' argument allows you to specify the number of outcome categories (top categories) with the default 10 elements. The top categories are the most frequent categories appearing the data. The '--chunk_size' argument allows you to specify how many descriptions that should be chunked together. The default is 10. '--test_size' allows you to specify the size of the test data with a default of 0.25. With the argument '--embedding_dim' you can change the default number of embedding dimensions from 50. '--epochs' allows you to specify the number of epochs with a default of 5. 

The last argument '--predict' is required and can be specified as either 'grape' or 'province'. The argument lets you specify what you are interested in predicting, either the grapes or the provinces of the wines. Please write the argument without quotations.

```bash
python wine.py --number 10 --chunk_size 10 --test_size 0.25 --embedding_dim 50 --epochs 10 --predict grape/province 
```
While running, status updates will be printed to the terminal. Afterwards, the results and plots can be found in the folder called 'out'. It takes approximately 4 minutes.