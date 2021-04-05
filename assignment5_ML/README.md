# Assignment 5 - LDA

## Research Statement 
__Unsupervised machine learning on philosophical texts__ <br>

I was interested in whether dominant topics in text from a large corpora of philosophical documents cluster according to the time period from which they originate. More specifically, by performing topic modelling I wanted to investigate if the dominance of different topics changes over time.

## Results 
By visually inspecting the the PCA plot, there does not seem to be a clear tendency of clustering according to the year from which the documents originate. This plot is a 2D feature reduced visualization and the 2 components are tehrefore very difficult to meaningfully interpret. The scatterplot indicates that certain topics could be more prominent in some centuries compared to others. However, analysing the corpora with 5 topics is an arbitrary choice of mine and the impression of the plot could have been dramatically different with a different number of predefined topics.

The plots and a csv file with the most prominent keywords for each topic can be found in the folder 'out'.


## Run the scripts
Setup the virtual environment and activate it
```bash
git clone https://github.com/askesvane/LanguageAnalytics.git
cd assignment5_ML
bash ./create_lang_venv.sh
source ./LDA_environment/bin/activate
```
Run the script (You can specify a subset of the data. The default is 50,000.
```bash
python LDA.py -subset 50000
```
