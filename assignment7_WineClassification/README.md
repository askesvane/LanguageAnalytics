# Assignment 7  

## Description of the assignment

The assignment is set to build a classification model which can predict the variety of grapes used to produce certain wine type based on its flavour description. 


__Background__

It is commonly known that the art of winemaking is an inseparable element of cultural identity for many nations around the world and that wine is one of the most frequently consumed alcoholic beverages. With the multitude of data on wine available online, it is possible to get a deeper insight into the field by building real-world tools helpful in the process of e.g., classifying the wine types or predicting the region of origin based on several wine characteristics. 
The creation of such classification tool is of a main focus of the current assignment where the type of grapes used in the wine production process and the region of their origin are predicted based on the linguistic flavor descriptions given by professional sommeliers. 
A working classification model brings about a possibility of creating a real-life, user-friendly tool that could be used in commercial contexts, e.g., in a creation of an interactive tool where the most suitable wine type is suggested based on user’s preferred taste notes or where the wine growing region is estimated based on the descriptive information on taste. 

The two main questions asked in the current assignment are therefore:


> Q1: Can the variety of grapes used to produce given wine be predicted from the linguistic description of its flavor characteristics?


> Q2: Can the province of a given wine origin be predicted from the linguistic description of its flavor characteristics?


___Data___

Data used in the current study was obtained from an online service for data scientist’s community- Kaggle.com. The acquired files took a form of .csv data file with 130k entries generated by a web scrape of a popular wine-rating service – WineEnthusiast.com.

Each entry contained information about:
- the specific country that the wine came from
- verbal descriptions of its flavor characteristics
- the name of the winery a specific bottle came from
- the number of points WineEnthusiast.com users rated the wine with (on a scale from 1-100)
- the price for a bottle(stated in the US dollars)
- the province/state and region the wine was from
- the title of the wine review
- the variety of the grape used in the winemaking process
- the name of the winery that produced a given wine

Data can be viewed and downloaded in the following link:
https://www.kaggle.com/zynicide/wine-reviews?select=winemag-data-130k-v2.csv



## The method
Before constructing the model, the imported data had to be preprocessed. As a first step, the data had to be chunked together according to the chunk size specified in the command line with the default size of 10. The data was subsequently balanced by wine variety or province so that all categories would be represented with the same amount of chunks. The script was designed in a was so that the classification outcomes, either the grape type or the province of wine origin can be specified by the user in the command line. 
Subsequently, the names of grape varieties or provinces mentioned in the original descriptions were removed. That has been done in order to prevent faulty predictions, where the wine variety or a province were predicted with a great accuracy, solely because the predicted label was mentioned in the original text corpus several times. By removing them from the text, it was possible to construct the model which predicts the category label from other descriptive characteristics, such as e.g., flavor notes. Upon necessary preprocessing the data was divided into a train and a test dataset. 
With keras Tokenizer() the text strings were converted into numbers. The length of the corpus was defined to be 5.000 regardless of the actual length. The text chunks were afterwards 'padded' so they were of an equal length corresponding to the longest text chunk.
The numerical representations were then converted into a dense embedded representation which allowed for the context can be taken into consideration. The library GloVe (glove.6B.50d) was used to enhance the model as every word in the corpus was mapped onto pretrained embeddings from GloVe. Thus, the model incorporated pretrained weights rather than only learned new weights from the data. The number of embedding dimensions was by default set to 50 dimensions but can be manipulated from the command line.

__The model__

The data was first fed into an embedding layer with the pretrained GloVe weights of the embedding matrix. The embedding layer was followed by a convolution layer employing the 'ReLU' activation function and with L2 regularization of 0.0001. This was predicted to constrain the model to learn a more regular set of weights in order to prevent overfitting. Then followed a max pooling layer and two dense layers, the first with a 'ReLU' activation function and the second with a 'softmax' activation function with n potential outcomes (where n signifies the desired number of outcome categories, either provinces or grape varieties- both possible to specify from the command line). The model was then compiled using 'Adam' as the optimizer and 'categorical crossentropy' as the loss function parameter.



## Results and evaluation

__VARIETY__ 

Running with default parameters, the training accuracy was predicted to be 99% while the test accuracy was 95%. Overall, the model is performing exceptionally well at predicting the specific grape variety.

When inspecting the training accuracy in the graph (available in the output folder) it is possible to see that the model is slightly overfitting to the training data around the 3 epochs (see the plot ‘grape_TraininglossAccuracy.png’). 

__PROVINCE__


Running with default parameters, the training accuracy was predicted to be 99% while the test accuracy was 96%. Overall, the model is performing exceptionally well at predicting the specific province of wine origin.

When inspecting the training accuracy on the graph it is possible to see that the model is slightly overfitting to the training data (see the plot province_TraininglossAccuracy.jpg') what is indicated by the divergence of the training and validation accuracy as well as training and validation loss around 2nd epoch. 


All results and figures with the outcome graphs can be found in the 'out' folder in the GitHub repository for this assignment. 

## Repository structure and files
This repository has the following directory structure:

| Column | Description|
|--------|:-----------|
```out``` | Contains the outputs with visuals from running the script.
```wine.py```| The script to be executed from the terminal.
```create_visual_venv.sh``` | A bash script which automatically generates a new virtual environment 'Covid_env', and install all the packages contained within 'requirements.txt'
```requirements.txt``` | A list of packages along with the versions that are required.
```README.md``` | This readme file.
```data```| A folder with a.csv file containing information about wine and its descriptions.


## Usage (reproducing the results)

### Virtual environment
In order to run the script, one is required to set up the virtual environment with all necessary packages installed. Please clone the repo, navigate to the folder for this assignment, run the bash script to set up the environment, and lastly activate it. The following code should be executed from the terminal:

```bash
git clone https://github.com/askesvane/VisualAnalytics.git
cd assignment7_WineClassification
bash ./create_visual_venv.sh
source ./Wine_env/bin/activate
```

### Download the ```GloVe``` library

In order to run the script 'wine.py', one needs to download an unzip the pretrained weights of the 'glove' library. Please execute the following from the terminal while being in the folder 'assignment7_WineClassification' (the zipfile will take 3-4 minutes to download):

```
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip -q glove.6B.zip
```


### Execute the script 
Now, the script can be executed. You can specify the --n argument which allows you to specify the number of the outcome categories (top categories) in both wine variety and province of the wine origin, with the default n of 10 elements, the --c which allows you to specify the chunk size with the default size of 10, --t which allows to specify the size of the test data with a default of 0.25, --d argument which signifies the embedding dimensions with a default of 50, --e which allows you to specify the number of epochs with a default of 5, and lastly, the --p parameter which allows you to specify the desired category that you want to predict, either the wine grape variety or the province of wine origin which you can choose by typing –p grape (for wine variety) or –p province (for province of origin prediction).

```bash
python wine.py --n 10 --c 10 --t 0.25 --d 50 --e 10 --p grape 

```
While running, status updates will be printed to the terminal. Afterwards, the classification report and plots can be found in the folder called 'out'. It takes approximately 4 minutes.