"""
A NetworkX addon program to compute the Ollivier-Ricci curvature of a given NetworkX graph.

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
    Ni, C.-C., Lin, Y.-Y., Gao, J., Gu, X., & Saucan, E. 2015. "Ricci curvature of the Internet topology"
        (Vol. 26, pp. 2758-2766). Presented at the 2015 IEEE Conference on Computer Communications (INFOCOM), IEEE.
    Ni, C.-C., Lin, Y.-Y., Gao, J., and Gu, X. 2018. "Network Alignment by Discrete Ollivier-Ricci Flow", Graph Drawing 2018.
    Ni, C.-C., Lin, Y.-Y., Luo, F. and Gao, J. 2019. "Community Detection on Networks with Ricci Flow", Scientific Reports.
    Ollivier, Y. 2009. "Ricci curvature of Markov chains on metric spaces". Journal of Functional Analysis, 256(3), 810-864.

"""
import importlib
import math
import sys
import time

import cvxpy as cvx
import networkit as nk
import networkx as nx
import numpy as np
import ot
import psutil
import ray

from .util import *

EPSILON = 1e-7


class OllivierRicci:

    def __init__(self, G: nx.Graph, alpha=0.5, weight="weight", method="OTD",
                 base=math.e, exp_power=2, proc=psutil.cpu_count(logical=False), verbose="ERROR"):
        """
        A class to compute Ollivier-Ricci curvature for all nodes and edges in G.
        Node Ricci curvature is defined as the average of all it's adjacency edge.
        :param G: A NetworkX graph.
        :param alpha: The parameter for the discrete Ricci curvature, range from 0 ~ 1.
                      It means the share of mass to leave on the original node.
                      E.g. x -> y, alpha = 0.4 means 0.4 for x, 0.6 to evenly spread to x's nbr.
                      Default: 0.5
        :param weight: The edge weight used to compute Ricci curvature. Default: weight
        :param method: Transportation method, "OTD" for Optimal Transportation Distance (Default),
                                              "ATD" for Average Transportation Distance.
                                              "Sinkhorn" for OTD approximated Sinkhorn distance.
        :param base: Base variable for weight distribution. Default: math.e
        :param exp_power: Exponential power for weight distribution. Default: 0
        :param verbose: Verbose level: ["INFO","DEBUG","ERROR"].
                            "INFO": show only iteration process log.
                            "DEBUG": show all output logs.
                            "ERROR": only show log if error happened (Default).
        """
        self.G = G.copy()
        self.alpha = alpha
        self.weight = weight
        self.method = method
        self.base = base
        self.exp_power = exp_power
        self.proc = proc

        self.set_verbose(verbose)

        self.EPSILON = 1e-7  # to prevent divided by zero

        assert importlib.util.find_spec("ot"), \
            "Package POT: Python Optimal Transport is required for Sinkhorn distance."

    def set_verbose(self, verbose):
        set_verbose(verbose)

    def compute_ricci_curvature_edges(self, edge_list=None):
        """
        Compute Ricci curvature of given edge lists.

        :return: G: A NetworkX graph with Ricci Curvature with edge attribute "ricciCurvature"
        """

        # Construct nx and nk's node look up dict
        nx2nk_ndict, nk2nx_ndict = {}, {}
        for idx, n in enumerate(self.G.nodes()):
            nx2nk_ndict[n] = idx
            nk2nx_ndict[idx] = n

        # Construct edge_list in nk's index
        edge_list_nk = []
        for x, y in edge_list or self.G.edges():
            edge_list_nk.append((nx2nk_ndict[x], nx2nk_ndict[y]))

        # Start to compute edge Ricci curvature by ray
        t0 = time.time()

        # Push to ray's share memory
        _G = ray.put(nx.to_scipy_sparse_matrix(self.G, weight=self.weight))

        # Open multiple actors that initialized with preloaded G, and edgelist for parallel computing
        actors = [
            OllivierRicciActor.remote(_G, edge_list_nk, self.alpha, self.method, self.base, self.exp_power, self.proc,
                                      i)
            for i in range(self.proc)]
        result = ray.get([actor.run.remote() for actor in actors])

        # flatten results collected from ray
        flat_result = [item for sublist in result for item in sublist]

        # convert edge index from nk back to nx for final output
        output = [((nk2nx_ndict[x[0]], nk2nx_ndict[x[1]]), y) for x, y in flat_result]

        logger.info("%8f secs for Ricci curvature computation." % (time.time() - t0))
        return output

    def compute_ricci_curvature(self):
        """
        Compute Ricci curvature of edges and nodes.
        The node Ricci curvature is defined as the average of node's adjacency edges.
        """

        if not nx.get_edge_attributes(self.G, self.weight):
            print('Edge weight not detected in graph, use "weight" as edge weight.')
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][self.weight] = 1.0
        edge_ricci = self.compute_ricci_curvature_edges(list(self.G.edges()))
        # Assign edge Ricci curvature from result to graph G
        for (source, target), rc in edge_ricci:
            self.G[source][target]['ricciCurvature'] = rc

        # Compute node Ricci curvature
        for n in self.G.nodes():
            rc_sum = 0  # sum of the neighbor Ricci curvature
            if nx.degree(self.G, n) != 0:
                for nbr in self.G.neighbors(n):
                    if 'ricciCurvature' in self.G[n][nbr]:
                        rc_sum += self.G[n][nbr]['ricciCurvature']

                # Assign the node Ricci curvature to be the average of node's adjacency edges
                self.G.nodes[n]['ricciCurvature'] = rc_sum / nx.degree(self.G, n)
                logger.debug("node %d, Ricci Curvature = %f" % (n, self.G.nodes[n]['ricciCurvature']))

    def compute_ricci_flow(self, iterations=100, step=1, delta=1e-4, surgery=(lambda G, *args, **kwargs: G, 100)):
        """
        Compute the given Ricci flow metric of each edge of a given connected Networkx graph.

        :param iterations: Iterations to require Ricci flow metric
        :param step: step size for gradient decent process
        :param delta: process stop when difference of Ricci curvature is within delta
        :param surgery: A tuple of user define surgery function that will execute every certain iterations
        """

        if not nx.is_connected(self.G):
            logger.warning("Not connected graph detected, compute on the largest connected component instead.")
            self.G = nx.Graph(self.G.subgraph(max(nx.connected_components(self.G), key=len)))
        self.G.remove_edges_from(nx.selfloop_edges(self.G))

        logger.info("Number of nodes: %d" % self.G.number_of_nodes())
        logger.info("Number of edges: %d" % self.G.number_of_edges())

        # Set normalized weight to be the number of edges.
        normalized_weight = float(self.G.number_of_edges())

        # Start compute edge Ricci flow
        t0 = time.time()

        if nx.get_edge_attributes(self.G, "original_RC"):
            logger.warning("original_RC detected, continue to refine the ricci flow.")
        else:
            self.compute_ricci_curvature()

            for (v1, v2) in self.G.edges():
                self.G[v1][v2]["original_RC"] = self.G[v1][v2]["ricciCurvature"]

        # Start the Ricci flow process
        for i in range(iterations):
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][self.weight] -= step * (self.G[v1][v2]["ricciCurvature"]) * self.G[v1][v2][self.weight]

            # Do normalization on all weight to prevent weight expand to infinity
            w = nx.get_edge_attributes(self.G, self.weight)
            sumw = sum(w.values())
            for k, v in w.items():
                w[k] = w[k] * (normalized_weight / sumw)
            nx.set_edge_attributes(self.G, w, self.weight)
            logger.info(" === Ricci flow iteration %d === " % i)

            self.compute_ricci_curvature()

            rc = nx.get_edge_attributes(self.G, "ricciCurvature")
            diff = max(rc.values()) - min(rc.values())

            logger.info("Ricci curvature difference: %f" % diff)
            logger.info("max:%f, min:%f | maxw:%f, minw:%f" % (
                max(rc.values()), min(rc.values()), max(w.values()), min(w.values())))

            if diff < delta:
                logger.info("Ricci curvature converged, process terminated.")
                break

            # do surgery or any specific evaluation
            surgery_func, do_surgery = surgery
            if i != 0 and i % do_surgery == 0:
                self.G = surgery_func(self.G, self.weight)
                normalized_weight = float(self.G.number_of_edges())

            for n1, n2 in self.G.edges():
                logger.debug(n1, n2, self.G[n1][n2])

        print("\n%8f secs for Ricci flow computation." % (time.time() - t0))


def sparse2nk(A):
    """
    Convert scipy.sparse.csr_matrix into Networkit graph
    :param A: `scipy.sparse.csr_matrix` matrix
    :return: Networkit graph
    """
    # check if graph is weighted
    if A.data.max() == A.data.min() == 1:
        is_weighted = False
    else:
        is_weighted = True

    # check if graph is directed
    if (abs(A.T - A) > 1e-10).nnz == 0:  # if A is symmetry
        is_directed = False
    else:
        is_directed = True

    num_nodes = A.shape[0]

    nkG = nk.graph.Graph(num_nodes, weighted=is_weighted, directed=is_directed)

    for x, y in zip(A.nonzero()[0], A.nonzero()[1]):
        if x < y and not is_directed:
            pass
        else:
            nkG.addEdge(x, y, A[x, y]) if is_weighted else nkG.addEdge(x, y)

    return nkG


@ray.remote
class OllivierRicciActor:
    """
    An remote actor for ray, each actor have a batch of edges to compute.
    """

    def __init__(self, G, edge_list, alpha, method, base, exp_power, proc, i):

        if sys.platform == 'linux':
            psutil.Process().cpu_affinity([i])
        self.G = sparse2nk(G)
        self.alpha = alpha
        self.method = method
        self.base = base
        self.exp_power = exp_power

        def partition(lst, n):
            division = len(lst) / n
            return [lst[round(division * j):round(division * (j + 1))] for j in range(n)]

        self.batches = partition(edge_list, proc)[i]

    def _get_pairwise_sp(self, source, target):
        return nk.distance.BidirectionalDijkstra(self.G, source, target).run().getDistance()

    # Construct the density distributions on each node
    def _get_single_node_neighbors_distributions(self, node, neighbors, direction="successors"):

        # Get sum of distributions from x's all neighbors
        nbr_edge_weights = []
        if direction == "predecessors":
            for nbr in neighbors:
                nbr_edge_weights.append(self.base ** (-(self._get_pairwise_sp(nbr, node)) ** self.exp_power))
        else:  # successors
            for nbr in neighbors:
                nbr_edge_weights.append(self.base ** (-(self._get_pairwise_sp(node, nbr)) ** self.exp_power))

        nbr_edge_weight_sum = sum(nbr_edge_weights)
        if nbr_edge_weight_sum > EPSILON:
            result = [(1.0 - self.alpha) * w / nbr_edge_weight_sum for w in nbr_edge_weights]

        elif len(neighbors) == 0:
            return []
        else:
            logger.warning("Neighbor weight sum too small, list:", neighbors)
            result = [(1.0 - self.alpha) / len(neighbors)] * len(neighbors)
        result.append(self.alpha)
        return result

    def run(self):

        results = []

        edges = self.batches
        for (source, target) in edges:

            assert source != target, "Self loop is not allowed."  # to prevent self loop
            # If the weight of edge is too small, return 0 instead.
            if self.G.weight(source, target) < EPSILON:
                logger.warning("Zero weight edge detected for edge (%s,%s), return Ricci Curvature as 0 instead." %
                               (source, target))
                results.append(((source, target), 0))
                continue

            # compute transportation distance
            m = 1  # assign an initial cost
            assert self.method in ["OTD", "ATD", "Sinkhorn"], \
                'Method %s not found, support method:["OTD", "ATD", "Sinkhorn"]' % self.method

            # Append source and target node into weight distribution matrix x,y

            # source_nbr = [x for x in self.G.iterInNeighbors(source)]
            source_nbr = self.G.neighbors(source)  # TODO: tmp fix for networkit6.0
            target_nbr = self.G.neighbors(target)
            # target_nbr = [x for x in self.G.iterNeighbors(target)]

            # Distribute densities for source and source's neighbors as x
            if not source_nbr:
                x = [1]
            else:
                x = self._get_single_node_neighbors_distributions(source, source_nbr, "predecessors")
            source_nbr.append(source)

            # Distribute densities for target and target's neighbors as y
            if not target_nbr:
                y = [1]
            else:
                y = self._get_single_node_neighbors_distributions(target, target_nbr, "successors")
            target_nbr.append(target)

            # construct the cost dictionary from x to y
            d = np.zeros((len(x), len(y)))

            for i, src in enumerate(source_nbr):
                for j, dst in enumerate(target_nbr):
                    d[i][j] = self._get_pairwise_sp(src, dst)

            x = np.array([x]).T  # the mass that source neighborhood initially owned
            y = np.array([y]).T  # the mass that target neighborhood needs to received

            if self.method == "OTD":
                t0 = time.time()
                rho = cvx.Variable((len(y), len(x)))  # the transportation plan rho

                # objective function d(x,y) * rho * x, need to do element-wise multiply here
                obj = cvx.Minimize(cvx.sum(cvx.multiply(np.multiply(d.T, x.T), rho)))

                # \sigma_i rho_{ij}=[1,1,...,1]
                source_sum = cvx.sum(rho, axis=0, keepdims=True)
                constrains = [rho * x == y, source_sum == np.ones((1, (len(x)))), 0 <= rho, rho <= 1]
                prob = cvx.Problem(obj, constrains)

                m = prob.solve(solver="ECOS_BB")  # change solver here if you want
                # solve for optimal transportation cost

                logger.debug(
                    "%8f secs for cvxpy. \t#source_nbr: %d, #target_nbr: %d" % (time.time() - t0, len(x), len(y)))

            elif self.method == "ATD":
                t0 = time.time()

                share = (1.0 - self.alpha) / (len(source_nbr) * len(target_nbr))
                cost_nbr = 0
                cost_self = self.alpha * self._get_pairwise_sp(source, target)

                for i, s in enumerate(source_nbr):
                    for j, t in enumerate(target_nbr):
                        cost_nbr += self._get_pairwise_sp(s, t) * share

                m = cost_nbr + cost_self  # Average transportation cost

                logger.debug("%8f secs for avg trans. dist. \t#source_nbr: %d, #target_nbr: %d" % (time.time() - t0,
                                                                                                   len(source_nbr),
                                                                                                   len(target_nbr)))
            elif self.method == "Sinkhorn":
                t0 = time.time()
                m = ot.sinkhorn2(x, y, d, 1e-1, method='sinkhorn')[0]
                logger.debug(
                    "%8f secs for Sinkhorn. dist. \t#source_nbr: %d, #target_nbr: %d" % (
                        time.time() - t0, len(x), len(y)))
            # compute Ricci curvature: k=1-(m_{x,y})/d(x,y)
            result = 1 - (m / self._get_pairwise_sp(source, target))  # Divided by the length of d(i, j)
            logger.debug("Ricci curvature (%s,%s) = %f" % (source, target, result))

            results.append(((source, target), result))
        return results
