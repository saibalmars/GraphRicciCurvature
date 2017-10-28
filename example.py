import networkx as nx
from GraphRicciCurvature.OllivierRicci import ricciCurvature

# import an example NetworkX karate club graph
G=nx.karate_club_graph()

# compute the ricci curvature of the given graph G
G=ricciCurvature(G)
print("The Ollivier-Ricci curvature of edge (0,1) is %f"%G[0][1]["ricciCurvature"])