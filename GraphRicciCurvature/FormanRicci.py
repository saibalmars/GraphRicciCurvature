"""
A NetworkX addon program to compute the Forman-Ricci curvature of a given NetworkX graph.

Copyright 2018 Chien-Chun Ni

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Author:
    Chien-Chun Ni
    http://www3.cs.stonybrook.edu/~chni/

Reference:
    Forman. 2003. “Bochner’s Method for Cell Complexes and Combinatorial Ricci Curvature.”
        Discrete & Computational Geometry 29 (3). Springer-Verlag: 323–74.
    Sreejith, R. P., Karthikeyan Mohanraj, Jürgen Jost, Emil Saucan, and Areejit Samal. 2016.
        “Forman Curvature for Complex Networks.” Journal of Statistical Mechanics: Theory and Experiment 2016 (6).
        IOP Publishing: 063206.

"""
import networkx as nx
from .util import *


class FormanRicci:
    def __init__(self, G, verbose="ERROR"):
        """
        A class to compute Forman-Ricci curvature for all nodes and edges in G.
        :param G: A connected NetworkX graph, unweighted graph only now, edge weight will be ignored.
        :param verbose: Verbose level: ["INFO","DEBUG","ERROR"].
                            "INFO": show only iteration process log.
                            "DEBUG": show all output logs.
                            "ERROR": only show log if error happened (Default).
        """

        self.G = G.copy()
        set_verbose(verbose)

    def compute_ricci_curvature(self):
        """
         Compute Forman-ricci curvature for all nodes and edges in G.
             Node curvature is defined as the average of all it's adjacency edge.
         :param G: A connected NetworkX graph, unweighted graph only now, edge weight will be ignored.
         """

        # Edge Forman curvature
        for (v1, v2) in self.G.edges():
            if self.G.is_directed():
                v1_nbr = set(list(self.G.predecessors(v1)) + list(self.G.successors(v1)))
                v2_nbr = set(list(self.G.predecessors(v2)) + list(self.G.successors(v2)))
            else:
                v1_nbr = set(self.G.neighbors(v1))
                v1_nbr.remove(v2)
                v2_nbr = set(self.G.neighbors(v2))
                v2_nbr.remove(v1)
            face = v1_nbr & v2_nbr
            # G[v1][v2]["face"]=face
            prl_nbr = (v1_nbr | v2_nbr) - face
            # G[v1][v2]["prl_nbr"]=prl_nbr

            self.G[v1][v2]["formanCurvature"] = len(face) + 2 - len(prl_nbr)

            logger.debug("Source: %s, target: %d, Forman-Ricci curvature = %f  " % (
                v1, v2, self.G[v1][v2]["formanCurvature"]))

        # Node Forman curvature
        for n in self.G.nodes():
            fcsum = 0  # sum of the neighbor Forman curvature
            if self.G.degree(n) != 0:
                for nbr in self.G.neighbors(n):
                    if 'formanCurvature' in self.G[n][nbr]:
                        fcsum += self.G[n][nbr]['formanCurvature']

                # assign the node Forman curvature to be the average of node's adjacency edges
                self.G.nodes[n]['formanCurvature'] = fcsum / self.G.degree(n)

            logger.debug("node %d, Forman Curvature = %f" % (n, self.G.nodes[n]['formanCurvature']))
        print("Forman curvature computation done.")
