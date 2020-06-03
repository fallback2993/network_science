#-------------------------------------------------------------------------------------
# Python code for first tutorial of Network Science
# Utrecht University
#
# PLEASE IGNORE THE FIRST 5 LINES FOR NOW AND LOOK BELOW THEM 
#-------------------------------------------------------------------------------------

import networkx as nx
import matplotlib
matplotlib.use("TkAgg")		# fix to make sure plots actually display
from matplotlib import pyplot as plot

#-------------------------------------------------------------------------------------
# FIRST PART
#-------------------------------------------------------------------------------------

# Read a graph from a file
# the path name has to be in English. So Users instead of Gebruikers. Use the full path if possible
# '\t' means at each row of the txt file a tab is read as a split in the coordinates
# change the file to any graph file you want
path_to_networks = "data/"
G = nx.read_edgelist(path_to_networks + "protein.edgelist.txt", delimiter='\t', create_using=nx.Graph())

# use the following for directed graphs!
# G = nx.read_edgelist("networks/metabolic.edgelist.txt", delimiter='\t', create_using=nx.DiGraph())

# use the NetworkX library to compute further structural properties, such as the average degree, the diameter, 
# the average shortest path length, the global clustering coefficient, the average clustering coefficient,
# find local bridges, etc.
# the idea is to play around for a bit, understand Python and NetworkX, and gain some insight into the different networks

print(nx.info(G)) # print general information about the graph.

# In the documentation there are many algorithms available for using directly on the graph objects
# https://networkx.github.io/documentation/networkx-2.3/reference/algorithms/index.html

# Such as the connected_components method, which gives you all connected components of the graph
# Then you can compute the size of the largest component as follows:
largest_cc = 0
for C in nx.connected_components(G):
	if len(C) > largest_cc:
		largest_cc = len(C)
print("Size of the largest connected component:", largest_cc)

# Or, much cleaner:
largest_cc = max(nx.connected_components(G), key=len)
print("Size of the largest connected component:", len(largest_cc))


#-------------------------------------------------------------------------------------
# SECOND PART
#-------------------------------------------------------------------------------------

# Use the nx.fast_gnp_random_graph() function to compute a random graph
# https://networkx.github.io/documentation/networkx-2.3/reference/generated/networkx.generators.random_graphs.fast_gnp_random_graph.html
# Now compute the same measures as you did before and compare them
# It might be helpful to generate N random graphs with the same parameters and average your computed values.
# Randomness might give weird values ;)



