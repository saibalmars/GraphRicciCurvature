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
import time
import math
import ot
from multiprocessing import Pool, cpu_count
from .util import *

import cvxpy as cvx
import networkx as nx
import numpy as np

EPSILON = 1e-7  # to prevent divided by zero


def _get_all_pairs_shortest_path():
    """
    Pre-compute the all pair shortest paths of the assigned graph G
    """
    logger.info("Start to compute all pair shortest path.")
    # Construct the all pair shortest path lookup
    if importlib.util.find_spec("networkit") is not None:
        import networkit as nk
        t0 = time.time()
        Gk = nk.nxadapter.nx2nk(_G, weightAttr=_weight)
        apsp = nk.distance.APSP(Gk).run().getDistances()
        lengths = {}
        for i, n1 in enumerate(_G.nodes()):
            lengths[n1] = {}
            for j, n2 in enumerate(_G.nodes()):
                if apsp[i][j] < 1e300:  # to drop unreachable node
                    lengths[n1][n2] = apsp[i][j]
        logger.info("%8f secs for all pair by NetworKit." % (time.time() - t0))
        return lengths
    else:
        logger.warning("NetworKit not found, use NetworkX for all pair shortest path instead.")
        t0 = time.time()
        lengths = dict(nx.all_pairs_dijkstra_path_length(_G, weight=_weight))
        logger.info("%8f secs for all pair by NetworkX." % (time.time() - t0))
        return lengths


def _get_edge_density_distributions():
    """
    Pre-compute densities distribution for all edges.
    """

    logger.info("Start to compute all pair density distribution for graph.")
    densities = dict()

    t0 = time.time()

    # Construct the density distributions on each node
    def get_single_node_neighbors_distributions(neighbors, direction="successors"):

        # Get sum of distributions from x's all neighbors
        if direction == "predecessors":
            nbr_edge_weight_sum = sum([_base ** (-(_lengths[nbr][x]) ** _exp_power)
                                       for nbr in neighbors])
        else:
            nbr_edge_weight_sum = sum([_base ** (-(_lengths[x][nbr]) ** _exp_power)
                                       for nbr in neighbors])

        if nbr_edge_weight_sum > EPSILON:
            if direction == "predecessors":
                result = [(1.0 - _alpha) * (_base ** (-(_lengths[nbr][x]) ** _exp_power)) /
                          nbr_edge_weight_sum for nbr in neighbors]
            else:
                result = [(1.0 - _alpha) * (_base ** (-(_lengths[x][nbr]) ** _exp_power)) /
                          nbr_edge_weight_sum for nbr in neighbors]
        elif len(neighbors) == 0:
            return []
        else:
            logger.warning("Neighbor weight sum too small, list:", neighbors)
            result = [(1.0 - _alpha) / len(neighbors)] * len(neighbors)
        result.append(_alpha)
        return result

    if _G.is_directed():
        for x in _G.nodes():
            predecessors = get_single_node_neighbors_distributions(list(_G.predecessors(x)), "predecessors")
            successors = get_single_node_neighbors_distributions(list(_G.successors(x)))
            densities[x] = {"predecessors": predecessors, "successors": successors}
    else:
        for x in _G.nodes():
            densities[x] = get_single_node_neighbors_distributions(list(_G.neighbors(x)))

    logger.info("%8f secs for edge density distribution construction" % (time.time() - t0))
    return densities


def _distribute_densities(source, target):
    """
    Get the density distributions of source and target node, and the cost (all pair shortest paths) between
    all source's and target's neighbors.
    :param source: Source node
    :param target: Target node
    :return: (source's neighbors distributions, target's neighbors distributions, cost dictionary)
    """

    # Append source and target node into weight distribution matrix x,y
    source_nbr = list(_G.predecessors(source)) if _G.is_directed() else list(_G.neighbors(source))
    target_nbr = list(_G.successors(target)) if _G.is_directed() else list(_G.neighbors(target))

    # Distribute densities for source and source's neighbors as x
    if not source_nbr:
        source_nbr.append(source)
        x = [1]
    else:
        source_nbr.append(source)
        x = _densities[source]["predecessors"] if _G.is_directed() else _densities[source]

    # Distribute densities for target and target's neighbors as y
    if not target_nbr:
        target_nbr.append(target)
        y = [1]
    else:
        target_nbr.append(target)
        y = _densities[target]["successors"] if _G.is_directed() else _densities[target]

    # construct the cost dictionary from x to y
    d = np.zeros((len(x), len(y)))

    for i, src in enumerate(source_nbr):
        for j, dst in enumerate(target_nbr):
            assert dst in _lengths[src], \
                "Target node not in list, should not happened, pair (%d, %d)" % (src, dst)
            d[i][j] = _lengths[src][dst]

    x = np.array([x]).T  # the mass that source neighborhood initially owned
    y = np.array([y]).T  # the mass that target neighborhood needs to received

    return x, y, d


def _optimal_transportation_distance(x, y, d):
    """
    Compute the optimal transportation distance (OTD) of the given density distributions by cvxpy.
    :param x: Source's neighbors distributions
    :param y: Target's neighbors distributions
    :param d: Cost dictionary
    :return: Optimal transportation distance
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
    :param x: Source's neighbors distributions
    :param y: Target's neighbors distributions
    :param d: Cost dictionary
    :return: Sinkhorn distance
    """
    t0 = time.time()
    m = ot.sinkhorn2(x, y, d, 1e-1, method='sinkhorn')[0]
    logger.debug(
        "%8f secs for Sinkhorn. dist. \t#source_nbr: %d, #target_nbr: %d" % (time.time() - t0, len(x), len(y)))

    return m


def _average_transportation_distance(source, target):
    """
    Compute the average transportation distance (ATD) of the given density distributions.
    :param source: Source node
    :param target: Target node
    :return: Average transportation distance
    """

    t0 = time.time()
    source_nbr = list(_G.predecessors(source)) if _G.is_directed() else list(_G.neighbors(source))
    target_nbr = list(_G.successors(target)) if _G.is_directed() else list(_G.neighbors(target))

    share = (1.0 - _alpha) / (len(source_nbr) * len(target_nbr))
    cost_nbr = 0
    cost_self = _alpha * _lengths[source][target]

    for i, s in enumerate(source_nbr):
        for j, t in enumerate(target_nbr):
            assert t in _lengths[s], "Target node not in list, should not happened, pair (%d, %d)" % (s, t)
            cost_nbr += _lengths[s][t] * share

    m = cost_nbr + cost_self  # Average transportation cost

    logger.debug("%8f secs for avg trans. dist. \t#source_nbr: %d, #target_nbr: %d" % (time.time() - t0,
                                                                                       len(source_nbr),
                                                                                       len(target_nbr)))
    return m


def _compute_ricci_curvature_single_edge(source, target):
    """
    Ricci curvature computation process for a given single edge.

    :param source: The source node
    :param target: The target node
    :return: The Ricci curvature of given edge

    """

    assert source != target, "Self loop is not allowed."  # to prevent self loop

    # If the weight of edge is too small, return 0 instead.
    if _lengths[source][target] < EPSILON:
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
    result = 1 - (m / _lengths[source][target])  # Divided by the length of d(i, j)
    logger.debug("Ricci curvature (%s,%s) = %f" % (source, target, result))

    return {(source, target): result}


def _compute_ricci_curvature_edges(G: nx.Graph(), alpha=0.5, weight="weight", method="OTD",
                                   base=math.e, exp_power=2, proc=cpu_count(), edge_list=None):
    """
    Compute Ricci curvature of given edge lists.

    :return: G: A NetworkX graph with Ricci Curvature with edge attribute "ricciCurvature"
    """

    if not edge_list:
        edge_list = []

    # ---set to global variable for multiprocessing used.---
    global _G
    global _alpha
    global _weight
    global _method
    global _base
    global _exp_power
    global _proc
    global _lengths
    global _densities
    # -------------------------------------------------------

    _G = G
    _alpha = alpha
    _weight = weight
    _method = method
    _base = base
    _exp_power = exp_power
    _proc = proc

    # Construct the all pair shortest path dictionary

    _lengths = _get_all_pairs_shortest_path()

    # Construct the density distribution

    _densities = _get_edge_density_distributions()

    # Start compute edge Ricci curvature
    t0 = time.time()

    p = Pool(processes=_proc)

    # Compute Ricci curvature for edges
    args = [(source, target) for source, target in edge_list]

    result = p.starmap_async(_compute_ricci_curvature_single_edge, args).get()
    p.close()
    p.join()
    logger.info("%8f secs for Ricci curvature computation." % (time.time() - t0))

    return result


def _compute_ricci_curvature(G: nx.Graph(), alpha=0.5, weight="weight", method="OTD",
                             base=math.e, exp_power=2, proc=cpu_count()):
    """
    Compute Ricci curvature of edges and nodes.
    The node Ricci curvature is defined as the average of node's adjacency edges.
    """

    if not nx.get_edge_attributes(G, weight):
        print('Edge weight not detected in graph, use "weight" as edge weight.')
        for (v1, v2) in G.edges():
            G[v1][v2][weight] = 1.0

    edge_ricci = _compute_ricci_curvature_edges(G, alpha=alpha, weight=weight, method=method,
                                                base=base, exp_power=exp_power, proc=proc, edge_list=G.edges())

    # Assign edge Ricci curvature from result to graph G
    for rc in edge_ricci:
        for k in list(rc.keys()):
            source, target = k
            G[source][target]['ricciCurvature'] = rc[k]

    # Compute node Ricci curvature
    for n in G.nodes():
        rc_sum = 0  # sum of the neighbor Ricci curvature
        if G.degree(n) != 0:
            for nbr in G.neighbors(n):
                if 'ricciCurvature' in G[n][nbr]:
                    rc_sum += G[n][nbr]['ricciCurvature']

            # Assign the node Ricci curvature to be the average of node's adjacency edges
            G.node[n]['ricciCurvature'] = rc_sum / G.degree(n)
            logger.debug("node %d, Ricci Curvature = %f" % (n, G.node[n]['ricciCurvature']))

    return G


def _compute_ricci_flow(G: nx.Graph(), alpha=0.5, weight="weight", method="OTD",
                        base=math.e, exp_power=2, proc=cpu_count(),
                        iterations=100, step=1, delta=1e-4, surgery=(lambda G, *args, **kwargs: G, 100)):
    """
    Compute the given Ricci flow metric of each edge of a given connected Networkx graph.

    :param iterations: Iterations to require Ricci flow metric
    :param step: step size for gradient decent process
    :param delta: process stop when difference of Ricci curvature is within delta
    :param surgery: A tuple of user define surgery function that will execute every certain iterations
    """

    if not nx.is_connected(G):
        logger.warning("Not connected graph detected, compute on the largest connected component instead.")
        G = nx.Graph(max(nx.connected_component_subgraphs(G), key=len))
    G.remove_edges_from(G.selfloop_edges())

    logger.info("Number of nodes: %d" % G.number_of_nodes())
    logger.info("Number of edges: %d" % G.number_of_edges())

    # Set normalized weight to be the number of edges.
    normalized_weight = float(G.number_of_edges())

    # Start compute edge Ricci flow
    t0 = time.time()

    if nx.get_edge_attributes(G, "original_RC"):
        logger.warning("original_RC detected, continue to refine the ricci flow.")
    else:
        _compute_ricci_curvature(G, alpha=alpha, weight=weight, method=method, base=base, exp_power=exp_power,
                                 proc=proc)

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
        nx.set_edge_attributes(G, w, weight)
        logger.info(" === Ricci flow iteration %d === " % i)

        _compute_ricci_curvature(G, alpha=alpha, weight=weight, method=method, base=base, exp_power=exp_power,
                                 proc=proc)

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

        # clear the APSP and densities since the graph have changed.
        global _lengths
        global _densities
        _lengths = {}
        _densities = {}

    logger.info("\n%8f secs for Ricci flow computation." % (time.time() - t0))

    return G


class OllivierRicci:

    def __init__(self, G, alpha=0.5, weight="weight", method="OTD",
                 base=math.e, exp_power=2, proc=cpu_count(), verbose="ERROR"):
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
        self.lengths = {}  # all pair shortest path dictionary
        self.densities = {}  # density distribution dictionary

        assert importlib.util.find_spec("ot"), \
            "Package POT: Python Optimal Transport is required for Sinkhorn distance."

    def set_verbose(self, verbose):
        set_verbose(verbose)

    def compute_ricci_curvature_edges(self, edge_list=None):
        return _compute_ricci_curvature_edges(G=self.G, alpha=self.alpha, weight=self.weight, method=self.method,
                                              base=self.base, exp_power=self.exp_power, proc=self.proc,
                                              edge_list=edge_list)

    def compute_ricci_curvature(self):
        return _compute_ricci_curvature(G=self.G, alpha=self.alpha, weight=self.weight, method=self.method,
                                        base=self.base, exp_power=self.exp_power, proc=self.proc)

    def compute_ricci_flow(self, iterations=100, step=1, delta=1e-4, surgery=(lambda G, *args, **kwargs: G, 100)):
        return _compute_ricci_flow(G=self.G, alpha=self.alpha, weight=self.weight, method=self.method,
                                   base=self.base, exp_power=self.exp_power, proc=self.proc,
                                   iterations=iterations, step=step, delta=delta, surgery=surgery)
