import networkx as nx
from GraphRicciCurvature.OllivierRicci import ricciCurvature

# import an example NetworkX karate club graph
G=nx.karate_club_graph()

# compute the ricci curvature of the given graph G
G=ricciCurvature(G)
print G[0][1]["ricciCurvature"]