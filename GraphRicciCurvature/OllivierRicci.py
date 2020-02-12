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
import time
import warnings
from multiprocessing import Pool, cpu_count

import cvxpy as cvx
import networkit as nk
import networkx as nx
import numpy as np
import ot

from .util import *

EPSILON = 1e-7  # to prevent divided by zero

# ---Shared global variables for multiprocessing used.---
_Gk = nk.graph.Graph()
_alpha = 0.5
_weight = "weight"
_method = "Sinkhorn"
_base = math.e
_exp_power = 2
_proc = cpu_count()

# -------------------------------------------------------


def _distribute_densities(source, target):
    """
    Get the density distributions of source and target node, and the cost (all pair shortest paths) between
    all source's and target's neighbors.

    :param source: Source node.
    :param target: Target node.
    :return: (source's neighbors distributions, target's neighbors distributions, cost dictionary).
    """

    # Append source and target node into weight distribution matrix x,y
    if not _Gk.isDirected():  # TODO: tmp fix for networkit6.0
        source_nbr = _Gk.neighbors(source)
        target_nbr = _Gk.neighbors(target)
    else:
        source_nbr = [x for x in _Gk.iterInNeighbors(source)]
        target_nbr = [x for x in _Gk.iterNeighbors(target)]

    def _get_single_node_neighbors_distributions(node, neighbors, direction="successors"):
        # Get sum of distributions from x's all neighbors
        if direction == "predecessors":
            nbr_edge_weights = [_base ** (-_get_pairwise_sp(nbr, node) ** _exp_power) for nbr in neighbors]
        else:  # successors
            nbr_edge_weights = [_base ** (-_get_pairwise_sp(node, nbr) ** _exp_power) for nbr in neighbors]

        nbr_edge_weight_sum = sum(nbr_edge_weights)
        if nbr_edge_weight_sum > EPSILON:
            # Sum need to be not too small to prevent divided by zero
            result = [(1.0 - _alpha) * w / nbr_edge_weight_sum for w in nbr_edge_weights]
        elif len(neighbors) == 0:
            # No neighbor, return [] instead
            return []
        else:
            logger.warning("Neighbor weight sum too small, list:", neighbors)
            result = [(1.0 - _alpha) / len(neighbors)] * len(neighbors)

        # Attach the _alpha share of distribution for node
        result.append(_alpha)
        return result

    # Distribute densities for source and source's neighbors as x
    x = _get_single_node_neighbors_distributions(source, source_nbr, "predecessors") if source_nbr else [1]
    source_nbr.append(source)

    # Distribute densities for target and target's neighbors as y
    y = _get_single_node_neighbors_distributions(target, target_nbr, "successors") if target_nbr else [1]
    target_nbr.append(target)

    # construct the cost dictionary from x to y
    d = np.zeros((len(x), len(y)))

    for i, src in enumerate(source_nbr):
        for j, tgt in enumerate(target_nbr):
            d[i][j] = _get_pairwise_sp(src, tgt)

    x = np.array([x]).T  # the mass that source neighborhood initially owned
    y = np.array([y]).T  # the mass that target neighborhood needs to received

    return x, y, d


def _get_pairwise_sp(source, target):
    """
    Compute pairwise shortest path from "source" to "target" by BidirectionalDijkstra via networkit

    :param source: Source node in networkit index.
    :param target: Target noe in networkit index.
    :return: pairwise shortest path length.
    """

    length = nk.distance.BidirectionalDijkstra(_Gk, source, target).run().getDistance()
    assert length < 1e300, "Shortest path between %d, %d is not found" % (source, target)
    return length


def _optimal_transportation_distance(x, y, d):
    """
    Compute the optimal transportation distance (OTD) of the given density distributions by cvxpy.

    :param x: Source's neighbors distributions.
    :param y: Target's neighbors distributions.
    :param d: Cost dictionary.
    :return: Optimal transportation distance.
    """

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

    logger.debug("%8f secs for cvxpy. \t#source_nbr: %d, #target_nbr: %d" % (time.time() - t0, len(x), len(y)))

    return m


def _sinkhorn_distance(x, y, d):
    """
    Compute the approximate optimal transportation distance (Sinkhorn distance) of the given density distributions.

    :param x: Source's neighbors distributions.
    :param y: Target's neighbors distributions.
    :param d: Cost dictionary.
    :return: Sinkhorn distance.
    """
    t0 = time.time()
    m = ot.sinkhorn2(x, y, d, 1e-1, method='sinkhorn')[0]
    logger.debug(
        "%8f secs for Sinkhorn. dist. \t#source_nbr: %d, #target_nbr: %d" % (time.time() - t0, len(x), len(y)))

    return m


def _average_transportation_distance(source, target):
    """
    Compute the average transportation distance (ATD) of the given density distributions.

    :param source: Source node.
    :param target: Target node.
    :return: Average transportation distance.
    """

    t0 = time.time()
    if not _Gk.isDirected():  # TODO: tmp fix for networkit6.0
        source_nbr = _Gk.neighbors(source)
        target_nbr = _Gk.neighbors(target)
    else:
        source_nbr = [x for x in _Gk.iterInNeighbors(source)]
        target_nbr = [x for x in _Gk.iterNeighbors(target)]

    share = (1.0 - _alpha) / (len(source_nbr) * len(target_nbr))
    cost_nbr = 0
    cost_self = _alpha * _get_pairwise_sp(source, target)

    for src in source_nbr:
        for tgt in target_nbr:
            cost_nbr += _get_pairwise_sp(src, tgt) * share

    m = cost_nbr + cost_self  # Average transportation cost

    logger.debug("%8f secs for avg trans. dist. \t#source_nbr: %d, #target_nbr: %d" % (time.time() - t0,
                                                                                       len(source_nbr),
                                                                                       len(target_nbr)))
    return m


def _compute_ricci_curvature_single_edge(source, target):
    """
    Ricci curvature computation process for a given single edge.

    :param source: The source node.
    :param target: The target node.
    :return: The Ricci curvature of given edge.
    """
    # print("EDGE:%s,%s"%(source,target))
    assert source != target, "Self loop is not allowed."  # to prevent self loop

    # If the weight of edge is too small, return 0 instead.
    if _Gk.weight(source, target) < EPSILON:
        logger.warning("Zero weight edge detected for edge (%s,%s), return Ricci Curvature as 0 instead." %
                       (source, target))
        return {(source, target): 0}

    # compute transportation distance
    m = 1  # assign an initial cost
    assert _method in ["OTD", "ATD", "Sinkhorn"], \
        'Method %s not found, support method:["OTD", "ATD", "Sinkhorn"]' % _method
    if _method == "OTD":
        x, y, d = _distribute_densities(source, target)
        m = _optimal_transportation_distance(x, y, d)
    elif _method == "ATD":
        m = _average_transportation_distance(source, target)
    elif _method == "Sinkhorn":
        x, y, d = _distribute_densities(source, target)
        m = _sinkhorn_distance(x, y, d)

    # compute Ricci curvature: k=1-(m_{x,y})/d(x,y)
    result = 1 - (m / _get_pairwise_sp(source, target))  # Divided by the length of d(i, j)
    logger.debug("Ricci curvature (%s,%s) = %f" % (source, target, result))

    return {(source, target): result}


def _wrap_compute_single_edge(stuff):
    """
    Wrapper for args in multiprocessing
    """
    return _compute_ricci_curvature_single_edge(*stuff)


def _compute_ricci_curvature_edges(G: nx.Graph, weight="weight", edge_list=[],
                                   alpha=0.5, method="OTD",
                                   base=math.e, exp_power=2, proc=cpu_count(), chunksize=None):
    """
    Compute Ricci curvature for edges in  given edge lists.

    :param G: A NetworkX graph.
    :param weight: The edge weight used to compute Ricci curvature. Default: "weight".
    :param edge_list: The list of edges to compute Ricci curvature, set to [] to run for all edges in G. Default: [].
    :param alpha: The parameter for the discrete Ricci curvature, range from 0 ~ 1.
                It means the share of mass to leave on the original node.
                E.g. x -> y, alpha = 0.4 means 0.4 for x, 0.6 to evenly spread to x's nbr.
                Default: 0.5.
    :param method: Transportation method, "OTD" for Optimal Transportation Distance (Default),
                                          "ATD" for Average Transportation Distance.
                                          "Sinkhorn" for OTD approximated Sinkhorn distance.
    :param base: Base variable for weight distribution. Default: math.e.
    :param exp_power: Exponential power for weight distribution. Default: 0.
    :param proc: Number of processor used for multiprocessing.
    :param chunksize: Chunk size for multiprocessing, set None for auto decide. Default: None.

    :return: output: A dictionary of edge Ricci curvature. E.g.: {(node1, node2): ricciCurvature}.
    """

    if not nx.get_edge_attributes(G, weight):
        print('Edge weight not detected in graph, use "weight" as default edge weight.')
        for (v1, v2) in G.edges():
            G[v1][v2][weight] = 1.0

    # ---set to global variable for multiprocessing used.---
    global _Gk
    global _alpha
    global _weight
    global _method
    global _base
    global _exp_power
    global _proc
    # -------------------------------------------------------

    _Gk = nk.nxadapter.nx2nk(G, weightAttr=weight)
    _alpha = alpha
    _weight = weight
    _method = method
    _base = base
    _exp_power = exp_power
    _proc = proc

    # Construct nx to nk dictionary
    nx2nk_ndict, nk2nx_ndict = {}, {}
    for idx, n in enumerate(G.nodes()):
        nx2nk_ndict[n] = idx
        nk2nx_ndict[idx] = n

    if edge_list:
        args = [(nx2nk_ndict[source], nx2nk_ndict[target]) for source, target in edge_list]
    else:
        args = [(nx2nk_ndict[source], nx2nk_ndict[target]) for source, target in G.edges()]

    # Start compute edge Ricci curvature
    t0 = time.time()

    p = Pool(processes=_proc)

    # Decide chunksize following method in map_async
    if chunksize is None:
        chunksize, extra = divmod(len(args), proc * 4)
        if extra:
            chunksize += 1

    # Compute Ricci curvature for edges
    result = p.imap_unordered(_wrap_compute_single_edge, args, chunksize=chunksize)
    p.close()
    p.join()

    # Convert edge index from nk back to nx for final output
    output = {}
    for rc in result:
        for k in list(rc.keys()):
            output[(nk2nx_ndict[k[0]], nk2nx_ndict[k[1]])] = rc[k]

    logger.info("%8f secs for Ricci curvature computation." % (time.time() - t0))

    return output


def _compute_ricci_curvature(G: nx.Graph, weight="weight", **kwargs):
    """
    Compute Ricci curvature of edges and nodes.
    The node Ricci curvature is defined as the average of node's adjacency edges.

    :param G: A NetworkX graph.
    :param weight: The edge weight used to compute Ricci curvature. Default: "weight".
    :return G: A networkx graph with "ricciCurvature" on nodes and edges.

    """

    if not nx.get_edge_attributes(G, weight):
        print('Edge weight not detected in graph, use "weight" as default edge weight.')
        for (v1, v2) in G.edges():
            G[v1][v2][weight] = 1.0

    # compute Ricci curvature for all edges
    edge_ricci = _compute_ricci_curvature_edges(G, weight=weight, **kwargs)

    # Assign edge Ricci curvature from result to graph G
    nx.set_edge_attributes(G, edge_ricci, "ricciCurvature")

    # Compute node Ricci curvature
    for n in G.nodes():
        rc_sum = 0  # sum of the neighbor Ricci curvature
        if G.degree(n) != 0:
            for nbr in G.neighbors(n):
                if 'ricciCurvature' in G[n][nbr]:
                    rc_sum += G[n][nbr]['ricciCurvature']

            # Assign the node Ricci curvature to be the average of node's adjacency edges
            G.nodes[n]['ricciCurvature'] = rc_sum / G.degree(n)
            logger.debug("node %d, Ricci Curvature = %f" % (n, G.nodes[n]['ricciCurvature']))

    return G


def _compute_ricci_flow(G: nx.Graph, weight="weight",
                        iterations=100, step=1, delta=1e-4, surgery=(lambda G, *args, **kwargs: G, 100),
                        **kwargs
                        ):
    """
    Compute the given Ricci flow metric of each edge of a given connected NetworkX graph.

    :param iterations: Iterations to require Ricci flow metric.
    :param step: step size for gradient decent process.
    :param delta: process stop when difference of Ricci curvature is within delta.
    :param surgery: A tuple of user define surgery function that will execute every certain iterations.
    :return: G: A NetworkX graph with weight as Ricci flow metric.
    """

    if not nx.is_connected(G):
        logger.warning("Not connected graph detected, compute on the largest connected component instead.")
        G = nx.Graph(G.subgraph(max(nx.connected_components(G), key=len)))
    G.remove_edges_from(nx.selfloop_edges(G))

    logger.info("Number of nodes: %d" % G.number_of_nodes())
    logger.info("Number of edges: %d" % G.number_of_edges())

    # Set normalized weight to be the number of edges.
    normalized_weight = float(G.number_of_edges())

    # Start compute edge Ricci flow
    t0 = time.time()

    if nx.get_edge_attributes(G, "original_RC"):
        logger.warning("original_RC detected, continue to refine the ricci flow.")
    else:
        _compute_ricci_curvature(G, weight=weight, **kwargs)

        for (v1, v2) in G.edges():
            G[v1][v2]["original_RC"] = G[v1][v2]["ricciCurvature"]

    # Start the Ricci flow process
    for i in range(iterations):
        for (v1, v2) in G.edges():
            G[v1][v2][weight] -= step * (G[v1][v2]["ricciCurvature"]) * G[v1][v2][weight]

        # Do normalization on all weight to prevent weight expand to infinity
        w = nx.get_edge_attributes(G, weight)
        sumw = sum(w.values())
        for k, v in w.items():
            w[k] = w[k] * (normalized_weight / sumw)
        nx.set_edge_attributes(G, values=w, name=weight)
        logger.info(" === Ricci flow iteration %d === " % i)

        _compute_ricci_curvature(G, weight=weight, **kwargs)

        rc = nx.get_edge_attributes(G, "ricciCurvature")
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
            G = surgery_func(G, weight)
            normalized_weight = float(G.number_of_edges())

        for n1, n2 in G.edges():
            logger.debug(n1, n2, G[n1][n2])

    logger.info("\n%8f secs for Ricci flow computation." % (time.time() - t0))

    return G


class OllivierRicci:

    def __init__(self, G: nx.Graph, weight="weight", alpha=0.5, method="OTD",
                 base=math.e, exp_power=2, proc=cpu_count(), chunksize=None, verbose="ERROR"):
        """
        A class to compute Ollivier-Ricci curvature for all nodes and edges in G.
        Node Ricci curvature is defined as the average of all it's adjacency edge.
        :param G: A NetworkX graph.
        :param weight: The edge weight used to compute Ricci curvature. Default: "weight".
        :param alpha: The parameter for the discrete Ricci curvature, range from 0 ~ 1.
              It means the share of mass to leave on the original node.
              E.g. x -> y, alpha = 0.4 means 0.4 for x, 0.6 to evenly spread to x's nbr.
              Default: 0.5.
        :param method: Transportation method, "OTD" for Optimal Transportation Distance (Default),
                                              "ATD" for Average Transportation Distance.
                                              "Sinkhorn" for OTD approximated Sinkhorn distance.
        :param base: Base variable for weight distribution. Default: math.e.
        :param exp_power: Exponential power for weight distribution. Default: 0.
        :param proc: Number of processor used for multiprocessing.
        :param chunksize: Chunk size for multiprocessing, set None for auto decide. Default: None.
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
        self.chunksize = chunksize

        self.set_verbose(verbose)
        self.lengths = {}  # all pair shortest path dictionary
        self.densities = {}  # density distribution dictionary

        if self.G.is_directed():
            warnings.warn("Directed graph might face some issue in this version. "
                          "Please use the previous version (0.3.1) via pip: "
                          "```pip3 install [--user] GraphRicciCurvature=0.3.1```. ")
        # assert not self.G.is_directed(), "Directed graph is not yet supported in this version."

        assert importlib.util.find_spec("ot"), \
            "Package POT: Python Optimal Transport is required for Sinkhorn distance."

    def set_verbose(self, verbose):
        """
        Set the verbose level for this process.

        :param verbose: Verbose level: ["INFO","DEBUG","ERROR"].
                            "INFO": show only iteration process log.
                            "DEBUG": show all output logs.
                            "ERROR": only show log if error happened (Default).
        """
        set_verbose(verbose)

    def compute_ricci_curvature_edges(self, edge_list=None):
        """
        Compute Ricci curvature for edges in  given edge lists.

        :param edge_list: The list of edges to compute Ricci curvature, set [] to run for all edges in G. Default: [].
        :return: output: A dictionary of edge Ricci curvature. E.g.: {(node1, node2): ricciCurvature}.
        """
        return _compute_ricci_curvature_edges(G=self.G, weight=self.weight, edge_list=edge_list,
                                              alpha=self.alpha, method=self.method,
                                              base=self.base, exp_power=self.exp_power,
                                              proc=self.proc, chunksize=self.chunksize)

    def compute_ricci_curvature(self):
        """
        Compute Ricci curvature of edges and nodes.
        The node Ricci curvature is defined as the average of node's adjacency edges.
        :return G: A networkx graph with "ricciCurvature" on nodes and edges.
        """

        self.G = _compute_ricci_curvature(G=self.G, weight=self.weight,
                                          alpha=self.alpha, method=self.method,
                                          base=self.base, exp_power=self.exp_power,
                                          proc=self.proc, chunksize=self.chunksize)
        return self.G

    def compute_ricci_flow(self, iterations=10, step=1, delta=1e-4, surgery=(lambda G, *args, **kwargs: G, 100)):
        """
        Compute the given Ricci flow metric of each edge of a given connected NetworkX graph.

        :param iterations: Iterations to require Ricci flow metric.
        :param step: step size for gradient decent process.
        :param delta: process stop when difference of Ricci curvature is within delta.
        :param surgery: A tuple of user define surgery function that will execute every certain iterations.
        :return: G: A NetworkX graph with "weight" as Ricci flow metric.
        """
        self.G = _compute_ricci_flow(G=self.G, weight=self.weight,
                                     iterations=iterations, step=step, delta=delta, surgery=surgery,
                                     alpha=self.alpha, method=self.method,
                                     base=self.base, exp_power=self.exp_power,
                                     proc=self.proc, chunksize=self.chunksize
                                     )
        return self.G
