#!/usr/bin/env python

#_____________# Import packages #_____________# 

# System tools
import os
import argparse

# Data handling
import pandas as pd

# drawing
import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)

#_____________# The script #_____________#


### FUNCTIONS

# function information
def information(nx_object, name_kind):
    """ 
    Function to gather information from an nx object and store it in a pandas df that is returned.
    """
    df = pd.DataFrame(nx_object) 
    df.columns =['Name', name_kind]
    return(df)

# function network_info
def network_info(G):
    """
    Function that employs the information() function and gathers information on eigenvector, 
    betweenness, and degree about the network. The function saves the gathered information as 
    a csv-file in the folder 'output'.
    """
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
    
    return

# Function network_plot
def network_plot(layout, G):
    """
    Function that creates a network with the layout specified in the terminal. The network is saved in 
    the folder 'viz' and is named after the specified layout.
    """
    # Define network image filename and path
    network_filename = os.path.join('viz',f'network_{layout}.png')
         
    # Create a network with the layout options specified in the terminal (either random or shell)
    if (layout == "random"):
        # draw nx network plot
        nx.draw_random(G, 
                       with_labels = True,  
                       node_color='red', 
                       font_weight= 'bold', 
                       node_size=1000, 
                       edge_color = "grey")
        
    elif (layout == "shell"):
        # draw nx network plot
        nx.draw_shell(G, 
                      with_labels = True,  
                      node_color='red', 
                      font_weight = 'bold', 
                      node_size=1000, 
                      edge_color = "grey")
    
    # Save the network to the folder 'viz' with the layout option in the filename
    plt.savefig(network_filename, dpi=100, bbox_inches="tight")
    return

    
### MAIN FUNCTION
def main(args):
    
    # Import parameters specified in the terminal
    filename = args["filename"]
    threshold = args["threshold"]
    layout = args["layout"]
    
    # Import the file specified in the commandline as a pd.
    filepath = os.path.join("data",filename)
    data = pd.read_csv(filepath)
    
    # Filter based on the weight of the edges
    filtered = data[data['weight'] > threshold]
    
    # Print data reduction in the terminal
    print(f"With a threshold of {threshold}, the data has been reduced from {str(len(data))} to {str(len(filtered))} entries.")
       
    # Create network object
    G = nx.from_pandas_edgelist(filtered, 'nodeA', 'nodeB', ["weight"], create_using=nx.Graph())
    
    # Create a network with the layout specified in the terminal and save it in the 'viz' folder.
    network_plot(layout, G)
    
    # Create df with relevant network information on each node and save it as a csv file
    network_info(G)
    
    # Print information to the terminal
    print(f"The network with a '{layout}' layout has been saved in the folder 'viz'. Relevant network information has been saved in the folder 'output'.")     

    
# RUN THE MAIN FUNCTION
if __name__=="__main__":
    
    # Argument parser
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-f", "--filename", type = str, default = "edgelist_weighted.csv",
                    help = "Specify the filename of the data.")
    ap.add_argument("-t", "--threshold", type = int, default = 500,
                    help = "Specify the threshold.")
    ap.add_argument("-l", "--layout", type = str, required = True,
                    help = "Specify one of the two layout options 'shell' or 'random'.")

    # Parse arguments. args is now an object containing all arguments added through the terminal. 
    argument_parser = vars(ap.parse_args())
    
    # run main() function
    main(argument_parser)

