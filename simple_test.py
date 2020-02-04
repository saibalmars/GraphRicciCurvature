import networkx as nx

from GraphRicciCurvature.OllivierRicci import OllivierRicci

# import an example NetworkX karate club graph
G = nx.karate_club_graph()

# compute the Ollivier-Ricci curvature of the given graph G
orc = OllivierRicci(G, alpha=0.5, method="Sinkhorn", verbose="INFO")
orc.compute_ricci_curvature()
print("Karate Club Graph: The Ollivier-Ricci curvature of edge (0,1) is %f" % orc.G[0][1]["ricciCurvature"])
