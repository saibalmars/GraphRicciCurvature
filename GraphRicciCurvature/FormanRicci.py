"""
A class to compute the Forman-Ricci curvature of a given NetworkX graph.
"""

# Author:
#     Chien-Chun Ni
#     http://www3.cs.stonybrook.edu/~chni/

# Reference:
#     Forman. 2003. “Bochner’s Method for Cell Complexes and Combinatorial Ricci Curvature.”
#         Discrete & Computational Geometry 29 (3). Springer-Verlag: 323–74.
#     Sreejith, R. P., Karthikeyan Mohanraj, Jürgen Jost, Emil Saucan, and Areejit Samal. 2016.
#         “Forman Curvature for Complex Networks.” Journal of Statistical Mechanics: Theory and Experiment 2016 (6).
#         IOP Publishing: 063206.
#     Samal, A., Sreejith, R.P., Gu, J. et al.
#         "Comparative analysis of two discretizations of Ricci curvature for complex networks."
#         Scientific Report 8, 8650 (2018).


import networkx as nx
import math
from .util import logger, set_verbose


class FormanRicci:
    def __init__(self, G: nx.Graph, weight="weight", method="augmented", verbose="ERROR"):
        """A class to compute Forman-Ricci curvature for all nodes and edges in G.

        Parameters
        ----------
        G : NetworkX graph
            A given NetworkX graph, unweighted graph only for now, edge weight will be ignored.
        weight : str
            The edge weight used to compute Ricci curvature. (Default value = "weight")
        method : {"1d", "augmented"}
            The method used to compute Forman-Ricci curvature. (Default value = "augmented")

            - "1d": Computed with 1-dimensional simplicial complex (vertex, edge).
            - "augmented": Computed with 2-dimensional simplicial complex, length <=3 (vertex, edge, face).
        verbose: {"INFO","DEBUG","ERROR"}
            Verbose level. (Default value = "ERROR")
                - "INFO": show only iteration process log.
                - "DEBUG": show all output logs.
                - "ERROR": only show log if error happened.
        """

        self.G = G.copy()
        self.weight = weight
        self.method = method

        if not nx.get_edge_attributes(self.G, self.weight):
            logger.info('Edge weight not detected in graph, use "weight" as default edge weight.')
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][self.weight] = 1.0
        if not nx.get_node_attributes(self.G, self.weight):
            logger.info('Node weight not detected in graph, use "weight" as default node weight.')
            for v in self.G.nodes():
                self.G.nodes[v][self.weight] = 1.0
        if self.G.is_directed():
            logger.info("Forman-Ricci curvature is not supported for directed graph yet, "
                        "covert input graph to undirected.")
            self.G = self.G.to_undirected()

        set_verbose(verbose)

    def compute_ricci_curvature(self):
        """Compute Forman-ricci curvature for all nodes and edges in G.
        Node curvature is defined as the average of all it's adjacency edge.

        Returns
        -------
        G: NetworkX graph
            A NetworkX graph with "formanCurvature" on nodes and edges.

        Examples
        --------
        To compute the Forman-Ricci curvature for karate club graph:

            >>> G = nx.karate_club_graph()
            >>> frc = FormanRicci(G)
            >>> frc.compute_ricci_curvature()
            >>> frc.G[0][2]
            {'weight': 1.0, 'formanCurvature': -7.0}
        """

        if self.method == "1d":
            # Edge Forman curvature
            for (v1, v2) in self.G.edges():
                v1_nbr = set(self.G.neighbors(v1))
                v1_nbr.remove(v2)
                v2_nbr = set(self.G.neighbors(v2))
                v2_nbr.remove(v1)

                w_e = self.G[v1][v2][self.weight]
                w_v1 = self.G.nodes[v1][self.weight]
                w_v2 = self.G.nodes[v2][self.weight]
                ev1_sum = sum([w_v1 / math.sqrt(w_e * self.G[v1][v][self.weight]) for v in v1_nbr])
                ev2_sum = sum([w_v2 / math.sqrt(w_e * self.G[v2][v][self.weight]) for v in v2_nbr])

                self.G[v1][v2]["formanCurvature"] = w_e * (w_v1 / w_e + w_v2 / w_e - (ev1_sum + ev2_sum))

                logger.debug("Source: %s, target: %d, Forman-Ricci curvature = %f  " % (
                    v1, v2, self.G[v1][v2]["formanCurvature"]))

        elif self.method == "augmented":
            # Edge Forman curvature
            for (v1, v2) in self.G.edges():
                v1_nbr = set(self.G.neighbors(v1))
                v1_nbr.remove(v2)
                v2_nbr = set(self.G.neighbors(v2))
                v2_nbr.remove(v1)

                face = v1_nbr & v2_nbr
                # prl_nbr = (v1_nbr | v2_nbr) - face

                w_e = self.G[v1][v2][self.weight]
                w_f = 1  # Assume all face have weight 1
                w_v1 = self.G.nodes[v1][self.weight]
                w_v2 = self.G.nodes[v2][self.weight]

                sum_ef = sum([w_e / w_f for _ in face])
                sum_ve = sum([w_v1 / w_e + w_v2 / w_e])

                # sum_ehef = sum([math.sqrt(w_e*self.G[v1][v][self.weight])/w_f +
                #                 math.sqrt(w_e*self.G[v2][v][self.weight])/w_f
                #                 for v in face])
                sum_ehef = 0  # Always 0 for cycle = 3 case.
                sum_veeh = sum([w_v1 / math.sqrt(w_e * self.G[v1][v][self.weight]) for v in (v1_nbr - face)] +
                               [w_v2 / math.sqrt(w_e * self.G[v2][v][self.weight]) for v in (v2_nbr - face)])

                self.G[v1][v2]["formanCurvature"] = w_e * (sum_ef + sum_ve - math.fabs(sum_ehef - sum_veeh))

                logger.debug("Source: %s, target: %d, Forman-Ricci curvature = %f  " % (
                    v1, v2, self.G[v1][v2]["formanCurvature"]))

        else:
            assert True, 'Method %s not available. Support methods: {"1d","augmented"}' % self.method

        # Node Forman curvature
        for n in self.G.nodes():
            fcsum = 0  # sum of the neighbor Forman curvature
            if self.G.degree(n) != 0:
                for nbr in self.G.neighbors(n):
                    if 'formanCurvature' in self.G[n][nbr]:
                        fcsum += self.G[n][nbr]['formanCurvature']

                # assign the node Forman curvature to be the average of node's adjacency edges
                self.G.nodes[n]['formanCurvature'] = fcsum / self.G.degree(n)
            else:
                self.G.nodes[n]['formanCurvature'] = fcsum

            logger.debug("node %d, Forman Curvature = %f" % (n, self.G.nodes[n]['formanCurvature']))
        logger.debug("Forman curvature (%s) computation done." % self.method)
