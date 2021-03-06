# Importing packages
import os
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


# Network analysis using object-oriented framework
'''
Your script should be able to be run from the command line
It should take any weighted edgelist as an input, providing that edgelist is saved as a CSV with the column headers "nodeA", "nodeB"
For any given weighted edgelist given as an input, your script should be used to create a network visualization, which will be saved in a folder called viz.
It should also create a data frame showing the degree, betweenness, and eigenvector centrality for each node. It should save this as a CSV in a folder called output.
'''

# Defining main function 
def main(args):
    # Adding arguments that can be specified in command line
    edgefile = args.fn 
    n_edges = args.ne 

    NetworkAnalysis(edgefile = edgefile, n_edges = n_edges),  # Calling main class 

# Setting class 'CountFunctions'
class NetworkAnalysis:

    def __init__(self, edgefile, n_edges):

        data_dir = self.setting_data_directory() # Setting directory of input data 
        out_dir = self.setting_output_directory() # Setting directory of output plots

        self.edgefile = edgefile # Setting filename as the provided filename
        
        if self.edgefile is None: # If no filename is specified, use edges_df.csv as default file
            self.edgefile = "edges_df.csv"
        
        self.n_edges = n_edges # Setting number of nodes to be kept

        df = pd.read_csv(data_dir / f'{self.edgefile}')  # Read csv edgefile

        # Creating network graph using pre-defined graphing function
        graph = self.get_network_graph(edgefile=df, n_edges=self.n_edges, out_dir = out_dir)

        # Calculating nodes metrics and saving df
        self.get_centrality_df(graph, out_dir = out_dir)


    # Defining function for setting directory for the raw data
    def setting_data_directory(self):

        root_dir = Path.cwd()  # Setting root directory

        data_dir = root_dir / 'W6-Network-Analysis' / 'data'   # Setting data directory

        return data_dir


    # Defining function for setting directory for the output
    def setting_output_directory(self):

        root_dir = Path.cwd()  # Setting root directory

        out_dir = root_dir / 'W6-Network-Analysis' / 'output' # Setting output directory

        return out_dir
    

    # Defining function for generating plot of network
    '''
    Generates a network graph from an edgefile containing a specified number of edges
    Args:
        edgefile (pd datatframe): Edgefile in a pd data frame format containing the following columns: 'nodeA', 'nodeB', 'weight
        n_edges (integer): Number of edges to keep 
    Returns:
        graph: Final network graph

    '''
    def get_network_graph(self, edgefile, n_edges, out_dir):
        # Selecting only the desired number of edges with the highest weight scores 
        filtered_df = edgefile.sort_values('weight', ascending = False).head(n_edges) # Contrary to setting a min_edge_weight, this method can be applied to all datasets 
        
        # Deleting extra column generated by previous line 
        del filtered_df["Unnamed: 0"]

        # Generating network graph
        graph = nx.from_pandas_edgelist(filtered_df, 'nodeA', 'nodeB', ["weight"])        

        pos = nx.draw_shell(graph,
                            with_labels = True, 
                            width = filtered_df['weight']/2500, # Weighting the width of the edges by their weight
                            font_weight= 'bold', 
                            edge_cmap = plt.cm.hsv,
                            edge_color = filtered_df['weight'], # Changing line colour according to weight - unfortunately I couldn't add a colorbar but it still looks kinda cool
                            font_color = "#2A2925", 
                            font_size = 6,
                            node_color = "#BD7575",
                            node_size = 150)
        
        # Save graph
        graph_path = out_dir / "network.png" # Output path for graph
        plt.savefig(graph_path, dpi=300, bbox_inches="tight") # Saving graph

        return graph


    # Defining function for creating and saving df containing degree, betweenness, and eigenvector centrality scores for all nodes in the network
    def get_centrality_df(self, graph, out_dir):

        # Calculate metrics 
        degree = nx.degree_centrality(graph)

        betweenness = nx.betweenness_centrality(graph)

        eigenvector = nx.eigenvector_centrality(graph)

        # Creating dataframe
        centrality_df = pd.DataFrame({
            'nodes': list(degree.keys()),
            'degree': list(degree.values()),
            'betweenness': list(betweenness.values()),
            'eigenvector': list(eigenvector.values()),  
        })

        # Saving dataframe
        df_path = out_dir / "centrality_df.csv"  # Output path for df
        centrality_df.to_csv(df_path) # Saving the df as a csv file

# Executing main function when script is run
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--fn', 
                        metavar="Filename",
                        type=str,
                        help='The name of the input edgefile',
                        required=False)

    parser.add_argument('--ne',
                        metavar="Number of edges",
                        type=int,
                        help='The number of edges to keep in the network.',
                        required=False,
                        default=50)

    main(parser.parse_args())