#!/usr/bin/python

#_____________# Import packages #_____________#

# standard library
import sys,os,argparse
sys.path.append(os.path.join(".."))
from pprint import pprint

# data and nlp
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])

# visualisation
import pyLDAvis.gensim
#pyLDAvis.enable_notebook()
import seaborn as sns
from matplotlib import rcParams
# figure size in inches
rcParams['figure.figsize'] = 20,10
import matplotlib.pyplot as plot

# LDA tools
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import nltk

# Stopwords
#nltk.download('stopwords')
#stop_words = stopwords.words('english')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


import lda_utils
from sklearn.decomposition import PCA

# warnings
import logging, warnings
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

#_____________# The script #_____________# 

## FUNCTIONS

# Function to import the data and extract relevant columns
def get_data(subset):
    
    # Import data from filepath 
    filename = os.path.join("data", "philosophy_data.csv")
    DATA = pd.read_csv(filename)

    # Create a subset and select relevant columns in the data
    DATA = DATA.iloc[:,[0,1,2,5,8]].sample(subset)
    
    return(DATA)

# Function for preprocessing the data
def preprocess_data(DATA):
    
    """
    Each document has several entries because a line per sentence appears in the data. 
    I comprise all text for each document in the same cell reducing the number of rows
    in DATA to the number of documents.
    """

    gathered_text = list(DATA.groupby(['title'])['sentence_lowered'].apply(lambda x: ' '.join(x)).reset_index()["sentence_lowered"])

    # Only keep 1 row per document.
    DATA = DATA.drop_duplicates(subset = ["title"])
    
    # Add the new text column
    DATA["text"] = gathered_text

    # Drop sentence_lower column since it only contains one sentence and we now have the text column
    DATA = DATA.drop("sentence_lowered", axis = 1)

    # convert column with years to numeric
    DATA["original_publication_date"] = pd.to_numeric(DATA["original_publication_date"])
    
    # Reduce the dataset to only contain documents after 1600 (Since we only have 4 earlier documents, I consider them outlieres)
    DATA = DATA[DATA['original_publication_date'] >= 1600]
    
    return(DATA)

# Function that return an LDA model based on the data 
def create_LDA_model(DATA):
    
    # Building the bigram and trigram models
    bigram = gensim.models.Phrases(DATA["text"], min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[DATA["text"]], min_count = 1, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    data_processed = lda_utils.process_words(DATA["text"], nlp, bigram_mod, trigram_mod,
                                             allowed_postags=['NOUN']) # consider: ['NOUN', "ADJ", "VERB", "ADV"]

    # Create Dictionary
    id2word = corpora.Dictionary(data_processed) # converts each word into an integer value (sort of like an ID)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_processed]

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics= 5, 
                                           random_state=100,
                                           chunksize=10,
                                           passes=10,
                                           iterations=100,
                                           per_word_topics=True, 
                                           minimum_probability=0.0)

    ## MODEL PERFORMANCE

    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, 
                                         texts=data_processed, 
                                         dictionary=id2word, 
                                         coherence='c_v')

    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    
    # Print topic content 
    pprint(lda_model.print_topics())
    
    return(lda_model, corpus, data_processed)


# Function that creates a new dataframe with the topic keywords
def df_topic_creator(lda_model, corpus, data_processed, DATA):

    # Creating a new DF, which assigns which topic is the most dominant, amongst all entries in the corpus
    df_topic_keywords = lda_utils.format_topics_sentences(ldamodel=lda_model, 
                                                          corpus=corpus, 
                                                          texts=data_processed)
    
    # titles, authors, schools, and year
    df_topic_keywords["Title"] = list(DATA["title"])
    df_topic_keywords["Author"] = list(DATA["author"])
    df_topic_keywords["School"] = list(DATA["school"])
    df_topic_keywords["Year"] = list(DATA["original_publication_date"])
    
    # Format
    df_topic_keywords.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text", "Title", "Author", "School", "Year"]

    return(df_topic_keywords)

# Function that adds the prevelances to the keyword df
def add_prevelance(df_topic_keywords, values):

    # Split topics and keep only values per topic
    split = []
    for entry in values:
        topic_prevelance = []
        for topic in entry:
            topic_prevelance.append(topic[1])
        split.append(topic_prevelance)
    
    # Save as df and combine with the topic keyword df
    split = pd.DataFrame(split, columns=["Topic1", "Topic2", "Topic3", "Topic4", "Topic5"])

    df_topic_keywords = pd.concat([df_topic_keywords, split], axis=1)
    
    return(df_topic_keywords)


# PCA feature reduction and plotting
def PCA_plotting(df_topic_keywords):
    # Reduce feature space 
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df_topic_keywords.set_index("Year").iloc[:,8:])
    principalDf = pd.DataFrame(data = principalComponents, 
                           columns = ['Principal Component 1', 'Principal Component 2'])

    # Adding Year
    principalDf["Year"] = df_topic_keywords["Year"]

    # I plot the PCA component scores of each document to see potential clusters spatially. Colored by year.
    plot_texts_pda = sns.relplot(
        data=principalDf,
        x="Principal Component 1", y="Principal Component 2", hue = "Year").fig.suptitle("2 Component Topic Model Space")

    # Saving to out
    fig = plot_texts_pda.get_figure()
    fig.savefig(os.path.join("out", "PCA_plot.png"))
    print("PCA figure has been successfully saved in the folder 'out' as PCA_plot.png")
    
    
    fig2 = df_topic_keywords.plot.scatter(x='Year', y='Topic_Num', title = "Scatter plot between showing the publishing year for each document and their respective most dominant topic")
    fig2.figure.savefig(os.path.join("out", "scatter_plot.png"))
    print("Scatterplot has been successfully saved in the folder 'out' as scatter_plot.png")
    

# Save as .csv file in the out folder 
def save_topics(df_topic_keywords):
    pd.set_option('max_colwidth', 90) # Making sure we can read all the topic keywords
    topic_keywords = df_topic_keywords.drop_duplicates(subset = ["Topic_Num"]).loc[:,["Topic_Num", "Keywords"]]
    topic_keywords.to_csv(os.path.join("out", "keywords_by_topics.csv"), index=False)
    print("Topics have been successfully saved in the folder 'out' as 'keywords_by_topics.csv'")


#________MAIN FUNCTION________#

def main(args):
    
    # Import parameters specified in the commanline
    subset = args["subset"]
    
    # Create dataframe
    DATA = get_data(subset)
    
    # Clean up the data
    DATA = preprocess_data(DATA)
    
    # create LDA model
    lda_model, corpus, data_processed = create_LDA_model(DATA)

    # Create a new dataframe with the output of the trained model
    df_topic_keywords = df_topic_creator(lda_model, corpus, data_processed, DATA)

    # Get the prevelance of each topic for every document
    values = list(lda_model.get_document_topics(corpus))
    df_topic_keywords = add_prevelance(df_topic_keywords, values)
    
    # Create a 2D plot (PCA reduced to 2 components) and save them
    PCA_plotting(df_topic_keywords)
    
    # Save topics
    save_topics(df_topic_keywords)


if __name__=="__main__":
    
    # Argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--subset", type = int, default = 50000,
                    help = "Specify a subset of the original data. Default is 50,000.")
    
    # Parse arguments. args is now an object containing all arguments added through the terminal. 
    argument_parser = vars(ap.parse_args())
    
    # run main() function
    main(argument_parser)












