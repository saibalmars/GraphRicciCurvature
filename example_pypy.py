from GraphRicciCurvature.OllivierRicci import *
import networkx as nx

G=nx.random_regular_graph(8,1000)
ricciCurvature(G)

# import numpy as np
# import ot
#
# n = 20  # nb samples
# xs = np.zeros((n, 2))
# xs[:, 0] = np.arange(n) + 1
# xs[:, 1] = (np.arange(n) + 1) * -0.001  # to make it strictly convex...
#
# xt = np.zeros((n, 2))
# xt[:, 1] = np.arange(n) + 1
#
# a, b = ot.unif(n), ot.unif(n)  # uniform distribution on samples
#
# # loss matrix
# M1 = ot.dist(xs, xt, metric='euclidean')
# M1 /= M1.max()
#
# # loss matrix
# M2 = ot.dist(xs, xt, metric='sqeuclidean')
# M2 /= M2.max()
#
# # loss matrix
# Mp = np.sqrt(ot.dist(xs, xt, metric='euclidean'))
# Mp /= Mp.max()