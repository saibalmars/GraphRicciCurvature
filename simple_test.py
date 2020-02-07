import networkx as nx
from memory_profiler import profile
from psutil import cpu_count

from GraphRicciCurvature.OllivierRicci import OllivierRicci

proc = cpu_count(logical=False)

# # import an example NetworkX karate club graph
# G = nx.karate_club_graph()
# # compute the Ollivier-Ricci curvature of the given graph G
# orc = OllivierRicci(G, alpha=0.5, method="Sinkhorn", verbose="DEBUG")
# orc.compute_ricci_curvature()

G_rr = nx.random_regular_graph(8, 3000)


@profile
def test():
    orc_rr = OllivierRicci(G_rr, proc=proc, verbose="INFO")
    orc_rr.compute_ricci_curvature()


test()
# cProfile.run("test()")
