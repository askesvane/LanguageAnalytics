

#_____________# Import packages #_____________# 

# System tools
import os

# Data analysis
import pandas as pd
from collections import Counter
from itertools import combinations 
from tqdm import tqdm
import argparse

# NLP
import spacy
nlp = spacy.load("en_core_web_sm")

# drawing
import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)

#_____________# The script #_____________# 

# Function using argparse to get a filename from the commandline.
def commandline_info():
    ap = argparse.ArgumentParser(description='Creating a network. Please provide a filename to an edgelist stored as a csv-file in the data folder as well as a filter.')
    ap.add_argument("-n", "--filename", required=True, type=str, help="str, the filename of any weighted edgelist stored as a csv-file")
    ap.add_argument("-t", "--threshold", required=False, type=int, default=500, help="int, a threshold filtering by the weight of the edges. Default = 500")
    
    args = vars(ap.parse_args())
    filename = args['filename']
    threshold = args['threshold']

    return(filename,threshold)

def information(nx_object, name_kind):
    df = pd.DataFrame(nx_object) 
    df.columns =['Name', name_kind]
    return(df)


def main():
    
    # Get the filename from the function filename()
    filename, threshold = commandline_info()
    #print(f"this is it {filename} and the treshold is {threshold}")

    # Import the file specified in the commandline as a pd.
    filepath = os.path.join("data",filename)
    data = pd.read_csv(filepath)
    
    # Filter based on the weight of the edges
    filtered = data[data["weight"] > threshold]
    
    
    # Define outpath 
    outpath_viz = os.path.join('viz',' network.png')
    
    # Create network
    G = nx.from_pandas_edgelist(filtered, 'nodeA', 'nodeB', ["weight"])
    nx.draw_shell(G, with_labels = True, font_weight= 'bold')
    plt.savefig(outpath_viz, dpi=300, bbox_inches="tight")
    
    
    # Create csv with relevant network information on each node
    
    # Eigenvector. Calculate eigenvector centrality and save it as a df with meaningful column names
    eigenvector = nx.eigenvector_centrality(G)
    eigenvector = information(eigenvector.items(), "eigenvector")
    
    # Betweenness. Calculate betweenness and save with meaningful column names.
    betweenness = nx.betweenness_centrality(G)
    betweenness = information(betweenness.items(), "betweenness")
    
    # Degree. Again df and meanignful column names
    degree = nx.degree(G)
    degree = information(degree, "degree")
    
    # Merge by shared column 'Name' to gather all relevant information
    network_info = eigenvector.merge(betweenness, on = 'Name').merge(degree, on = 'Name')
    
    # Save as 'network_info.csv' in the folder 'output'
    network_info.to_csv("output/network_info.csv",index=False)


if __name__=="__main__":
    main()










