# Assignment 4 - Network Analysis

__Contribution__<br>
The code of this assignment was created by me.

## Description of the assignment

__Creating a reusable network analysis pipeline__

Create a command line tool that will take a given dataset and perform a simple network analysis on it. In particular, it should build networks based on entities appearing together in the same documents, like we did in class.

- Your script should be able to be run from the command line
- It should take any weighted edgelist as an input, providing that edgelist is saved as a CSV with the column headers "nodeA", "nodeB"
- For any given weighted edgelist given as an input, your script should be used to create a network visualization, which will be saved in a folder called 'viz'.
- It should also create a data frame showing the degree, betweenness, and eigenvector centrality for each node. It should save this as a CSV in a folder called 'output'.

## Methods 

First, the edgelist imported is filtered by the weights of the edges between different nodes according to the threshold specified in the terminal. The default is 500.

Subsequently, the package 'networkx' is employed to construct a network object from the remaining data of the edgelist. The network object is used to create a graphical network visualization with one of the two possible layouts also specified in the terminal - either 'shell' or 'random'. The visualization is saved in the folder 'viz'.

Lastly, 'eigenvector centrality' and 'betweenness' as well as 'degree' are saved as a csv-file in the folder 'output'. The information is extracted from the network object created with 'networkx'.

## Results and evaluation

The command line tool I have coded can be used on any csv-file with the right column structure. I evaluate the network visualizations based on the test dataset 'edgelist_weighted.csv' provided in the data folder.

The two networks 'network_random.png' and 'network_shell.png' visualize the same data but with different layouts. In both networks, the importance of 'Clinton' is very prominent with many edges to nodes spread over the whole network. In addition, 'Bush' appears to be connecting with a significant amount of nodes as well. It is very clear that the two different layouts highly influence the interpretability of the networks: The visualization 'network_shell.png' is fairly interpretable and easy to assess compared to 'network_random.png' which appears confusing and messy with a chaotic web of edges spread out between most of the nodes. 

To enhance the interpretability of the network, varying the sizes of the nodes relative to the amount of edges would make it easier to quickly spot more important nodes. In addition, a layout that groups the nodes by their degree of interconnection would also provide a better overview over possible tendencies in the data.

## Repository structure and files
This repository has the following directory structure:

| Column | Description|
|--------|:-----------|
```data```| A folder with the test dataset 'edgelist_weighted.csv'.
```output```| A folder with the output csv-file from running the script.
```viz```| A folder with the visualizations generated when running the script. It contains a network visualization of the test dataset with two different layouts, 'shell' and 'random'.
```create_lang_venv.sh```| A bash script to set up the virtual environment.
```network.py```| The script to be executed from the command line.
```README.md```| This file.
```requirements.txt```| All packages required to run the script. They will automatically be installed in the virtual environment when running the bash script.

## Usage (reproducing the results)

### Virtual environment
In order to run the script, one is required to set up the virtual environment with all necessary packages installed. Please clone the repo, navigate to the folder for this assignment, run the bash script to set up the environment, and lastly activate it. The following code should be executed from the terminal:

```bash
git clone https://github.com/askesvane/LanguageAnalytics.git
cd assignment4_NetworkAnalysis
bash ./create_lang_venv.sh
source ./network_env/bin/activate
```

### Execute the script 
Now, the script can be executed. With the argument '--filename', one can select a csv-file on which the command line tool should be applied. The default is the filename of the provided test dataset 'edgelist_weighted.csv'. The argument '--threshold' can be used to specify the threshold of the weights of the edges. The default is 500. '--layout' is a required argument and must be either specified as 'shell' or 'random'. This is the layout of the network that will be generated when executing the script.

```bash
python network.py --filename 'edgelist_weighted.csv' --threshold 500 --layout shell/random
```
After running, the network can be found in the folder ```viz``` saved as a png-file. The csv-file with additional information can be found in the folder ```output```.



