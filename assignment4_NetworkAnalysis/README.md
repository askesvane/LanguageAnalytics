# Assignment 4 - Network Analysis

## Description of the assignment

Creating reusable network analysis pipeline

This exercise is building directly on the work we did in class. I want you to take the code we developed together and in you groups and turn it into a reusable command-line tool. You can see the code from class here: https://github.com/CDS-AU-DK/cds-language/blob/main/notebooks/session6.ipynb

This command-line tool will take a given dataset and perform simple network analysis. In particular, it will build networks based on entities appearing together in the same documents, like we did in class.

- Your script should be able to be run from the command line
- It should take any weighted edgelist as an input, providing that edgelist is saved as a CSV with the column headers "nodeA", "nodeB"
- For any given weighted edgelist given as an input, your script should be used to create a network visualization, which will be saved in a folder called viz.
- It should also create a data frame showing the degree, betweenness, and eigenvector centrality for each node. It should save this as a CSV in a folder called output.

## Feedback instructions

- Clone the whole repository to a chosen location on your computer or JupytorHub
- Through the terminal, navigate to the folder: ../LanguageAnalytics/assignment4_NetworkAnalysis

### Set up virtual environment

- Execute in the terminal: ```bash create_lang_venv.sh```. This will create a virtual environment called ```network_environment``` which will install everything in the ```requirements.txt``` file.
- Activate the virtual environment by executing ```source ./network_environment/bin/activate```. You should now see that the environment is activated at your command line.

### Run the .py script

- You are required to specify a file on which the script should run the analysis. Any file saved in the folder ```data``` as a .csv with the column headers "nodeA", "nodeB", and "weight" can be used. 
- You can specify a minimum threshold to filter out less significant edges. The default is 500. 
- A .csv file ```edgelist_weighted.csv``` is provided. From the commandline, execute ```python network.py -n edgelist_weighted.csv -t [TRESHOLD]```. Remember to specify a threshold or delete the flag.
- The network can be found in the folder ```viz``` saved as a .png file. The .csv file with additional information can be found in the folder ```output```.




