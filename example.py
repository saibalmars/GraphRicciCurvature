import networkx as nx

from GraphRicciCurvature.FormanRicci import formanCurvature
from GraphRicciCurvature.OllivierRicci import ricciCurvature

# import an example NetworkX karate club graph
G = nx.karate_club_graph()

# compute the Ollivier-Ricci curvature of the given graph G
G = ricciCurvature(G)
print("Karate Club Graph: The Ollivier-Ricci curvature of edge (0,1) is %f" % G[0][1]["ricciCurvature"])

# compute the Forman-Ricci curvature of the given graph G
G = formanCurvature(G)
print("Karate Club Graph: The Forman-Ricci curvature of edge (0,1) is %f" % G[0][1]["formanCurvature"])

# Construct a directed graph example
Gd = nx.DiGraph()
Gd.add_edges_from([(1, 2), (2, 3), (3, 4), (2, 4), (4, 2)])

# compute the Ollivier-Ricci curvature of the given directed graph Gd
Gd = ricciCurvature(Gd)
for n1, n2 in Gd.edges():
    print("Directed Graph: The Ollivier-Ricci curvature of edge(%d,%d) id %f" % (n1, n2, Gd[n1][n2]["ricciCurvature"]))

# compute the Forman-Ricci curvature of the given directed graph Gd
Gd = formanCurvature(Gd)
for n1, n2 in Gd.edges():
    print("Directed Graph: The Forman-Ricci curvature of edge(%d,%d) id %f" % (n1, n2, Gd[n1][n2]["formanCurvature"]))
