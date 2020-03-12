"""
A class to compute the Ollivier-Ricci curvature of a given NetworkX graph.
"""

# Author:
#     Chien-Chun Ni
#     http://www3.cs.stonybrook.edu/~chni/

# Reference:
#     Ni, C.-C., Lin, Y.-Y., Gao, J., Gu, X., & Saucan, E. 2015.
#         "Ricci curvature of the Internet topology" (Vol. 26, pp. 2758-2766).
#         Presented at the 2015 IEEE Conference on Computer Communications (INFOCOM), IEEE.
#     Ni, C.-C., Lin, Y.-Y., Gao, J., and Gu, X. 2018.
#         "Network Alignment by Discrete Ollivier-Ricci Flow", Graph Drawing 2018.
#     Ni, C.-C., Lin, Y.-Y., Luo, F. and Gao, J. 2019.
#         "Community Detection on Networks with Ricci Flow", Scientific Reports.
#     Ollivier, Y. 2009.
#         "Ricci curvature of Markov chains on metric spaces". Journal of Functional Analysis, 256(3), 810-864.


import heapq
import importlib
import math
import time
from functools import lru_cache
from multiprocessing import Pool, cpu_count

import cvxpy as cvx
import networkit as nk
import networkx as nx
import numpy as np
import ot

from .util import logger, set_verbose

EPSILON = 1e-7  # to prevent divided by zero

# ---Shared global variables for multiprocessing used.---
_Gk = nk.graph.Graph()
_alpha = 0.5
_weight = "weight"
_method = "Sinkhorn"
_base = math.e
_exp_power = 2
_proc = cpu_count()
_cache_maxsize = 1000000
_nbr_topk = 50

# -------------------------------------------------------


def _distribute_densities(source, target, nbr_topk=_nbr_topk):
    """Get the density distributions of source and target node, and the cost (all pair shortest paths) between
    all source's and target's neighbors.

    Parameters
    ----------
    source : int
        Source node index in Networkit graph `_Gk`.
    target : int
        Target node index in Networkit graph `_Gk`.
    nbr_topk : int
        Only take the neighbors with top k edge weights.
    Returns
    -------
    x : (m,) numpy.ndarray
        Source's density distributions, includes source and source's neighbors.
    y : (n,) numpy.ndarray
        Target's density distributions, includes source and source's neighbors.
    d : (m, n) numpy.ndarray
        Shortest path matrix.

    """

    # Append source and target node into weight distribution matrix x,y
    source_nbr = _Gk.inNeighbors(source)
    target_nbr = _Gk.neighbors(target)

    def _get_single_node_neighbors_distributions(node, neighbors, direction="successors"):
        # Get sum of distributions from x's all neighbors
        heap_weight_node_pair = []
        if direction == "predecessors":
            for nbr in neighbors:
                if len(heap_weight_node_pair) < nbr_topk:
                    heapq.heappush(heap_weight_node_pair, (_base ** (-_get_pairwise_sp(nbr, node) ** _exp_power), nbr))
                else:
                    heapq.heappushpop(heap_weight_node_pair, (_base ** (-_get_pairwise_sp(nbr, node) ** _exp_power), nbr))
        else:  # successors
            for nbr in neighbors:
                if len(heap_weight_node_pair) < nbr_topk:
                    heapq.heappush(heap_weight_node_pair, (_base ** (-_get_pairwise_sp(node, nbr) ** _exp_power), nbr))
                else:
                    heapq.heappushpop(heap_weight_node_pair, (_base ** (-_get_pairwise_sp(node, nbr) ** _exp_power), nbr))

        nbr_edge_weight_sum = sum([x[0] for x in heap_weight_node_pair])

        if nbr_edge_weight_sum > EPSILON:
            # Sum need to be not too small to prevent divided by zero
            distributions = [(1.0 - _alpha) * w / nbr_edge_weight_sum for w, _ in heap_weight_node_pair]
            nbr = [x[1] for x in heap_weight_node_pair]
            return distributions + [_alpha], nbr + [node]
        elif len(neighbors) == 0:
            # No neighbor, all mass stay at node
            return [1], [node]
        else:
            logger.warning("Neighbor weight sum too small, list:", heap_weight_node_pair)
            distributions = [(1.0 - _alpha) / len(heap_weight_node_pair)] * len(heap_weight_node_pair)
            nbr = [x[1] for x in heap_weight_node_pair]
            return distributions + [_alpha], nbr + [node]

    # Distribute densities for source and source's neighbors as x
    x, source_topknbr = _get_single_node_neighbors_distributions(source, source_nbr, "predecessors")

    # Distribute densities for target and target's neighbors as y
    y, target_topknbr = _get_single_node_neighbors_distributions(target, target_nbr, "successors")

    # construct the cost dictionary from x to y
    d = np.zeros((len(x), len(y)))

    for i, src in enumerate(source_topknbr):
        for j, tgt in enumerate(target_topknbr):
            d[i][j] = _get_pairwise_sp(src, tgt)

    x = np.array([x]).T  # the mass that source neighborhood initially owned
    y = np.array([y]).T  # the mass that target neighborhood needs to received

    return x, y, d


@lru_cache(_cache_maxsize)
def _get_pairwise_sp(source, target):
    """Compute pairwise shortest path from `source` to `target` by BidirectionalDijkstra via Networkit.

    Parameters
    ----------
    source : int
        Source node index in Networkit graph `_Gk`.
    target : int
        Target node index in Networkit graph `_Gk`.

    Returns
    -------
    length : float
        Pairwise shortest path length.

    """

    length = nk.distance.BidirectionalDijkstra(_Gk, source, target).run().getDistance()
    assert length < 1e300, "Shortest path between %d, %d is not found" % (source, target)
    return length


def _optimal_transportation_distance(x, y, d):
    """Compute the optimal transportation distance (OTD) of the given density distributions by CVXPY.

    Parameters
    ----------
    x : (m,) numpy.ndarray
        Source's density distributions, includes source and source's neighbors.
    y : (n,) numpy.ndarray
        Target's density distributions, includes source and source's neighbors.
    d : (m, n) numpy.ndarray
        Shortest path matrix.

    Returns
    -------
    m : float
        Optimal transportation distance.

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
    """Compute the approximate optimal transportation distance (Sinkhorn distance) of the given density distributions.

    Parameters
    ----------
    x : (m,) numpy.ndarray
        Source's density distributions, includes source and source's neighbors.
    y : (n,) numpy.ndarray
        Target's density distributions, includes source and source's neighbors.
    d : (m, n) numpy.ndarray
        Shortest path matrix.

    Returns
    -------
    m : float
        Sinkhorn distance, an approximate optimal transportation distance.

    """
    t0 = time.time()
    m = ot.sinkhorn2(x, y, d, 1e-1, method='sinkhorn')[0]
    logger.debug(
        "%8f secs for Sinkhorn. dist. \t#source_nbr: %d, #target_nbr: %d" % (time.time() - t0, len(x), len(y)))

    return m


def _average_transportation_distance(source, target):
    """Compute the average transportation distance (ATD) of the given density distributions.

    Parameters
    ----------
    source : int
        Source node index in Networkit graph `_Gk`.
    target : int
        Target node index in Networkit graph `_Gk`.

    Returns
    -------
    m : float
        Average transportation distance.

    """

    t0 = time.time()
    source_nbr = _Gk.inNeighbors(source)
    target_nbr = _Gk.neighbors(target)

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
    """Ricci curvature computation for a given single edge.

    Parameters
    ----------
    source : int
        Source node index in Networkit graph `_Gk`.
    target : int
        Target node index in Networkit graph `_Gk`.

    Returns
    -------
    result : dict[(int,int), float]
        The Ricci curvature of given edge in dict format. E.g.: {(node1, node2): ricciCurvature}

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
        x, y, d = _distribute_densities(source, target, _nbr_topk)
        m = _optimal_transportation_distance(x, y, d)
    elif _method == "ATD":
        m = _average_transportation_distance(source, target)
    elif _method == "Sinkhorn":
        x, y, d = _distribute_densities(source, target, _nbr_topk)
        m = _sinkhorn_distance(x, y, d)

    # compute Ricci curvature: k=1-(m_{x,y})/d(x,y)
    result = 1 - (m / _get_pairwise_sp(source, target))  # Divided by the length of d(i, j)
    logger.debug("Ricci curvature (%s,%s) = %f" % (source, target, result))

    return {(source, target): result}


def _wrap_compute_single_edge(stuff):
    """Wrapper for args in multiprocessing."""
    return _compute_ricci_curvature_single_edge(*stuff)


def _compute_ricci_curvature_edges(G: nx.Graph, weight="weight", edge_list=[],
                                   alpha=0.5, method="OTD",
                                   base=math.e, exp_power=2, proc=cpu_count(), chunksize=None, cache_maxsize=1000000,
                                   nbr_topk=1000):
    """Compute Ricci curvature for edges in  given edge lists.

    Parameters
    ----------
    G : NetworkX graph
        A given directional or undirectional NetworkX graph.
    weight : str
        The edge weight used to compute Ricci curvature. (Default value = "weight")
    edge_list : list of edges
        The list of edges to compute Ricci curvature, set to [] to run for all edges in G. (Default value = [])
    alpha : float
        The parameter for the discrete Ricci curvature, range from 0 ~ 1.
        It means the share of mass to leave on the original node.
        E.g. x -> y, alpha = 0.4 means 0.4 for x, 0.6 to evenly spread to x's nbr.
        (Default value = 0.5)
    method : {"OTD", "ATD", "Sinkhorn"}
        The optimal transportation distance computation method. (Default value = "OTD")

        Transportation method:
            - "OTD" for Optimal Transportation Distance,
            - "ATD" for Average Transportation Distance.
            - "Sinkhorn" for OTD approximated Sinkhorn distance.
    base : float
        Base variable for weight distribution. (Default value = `math.e`)
    exp_power : float
        Exponential power for weight distribution. (Default value = 0)
    proc : int
        Number of processor used for multiprocessing. (Default value = `cpu_count()`)
    chunksize : int
        Chunk size for multiprocessing, set None for auto decide. (Default value = `None`)
    cache_maxsize : int
        Max size for LRU cache for pairwise shortest path computation.
        Set this to `None` for unlimited cache. (Default value = 1000000)
    nbr_topk : int
        Only take the top k edge weight neighbors for density distribution.
        Smaller k run faster but the result is less accurate. (Default value = 1000)

    Returns
    -------
    output : dict[(int,int), float]
        A dictionary of edge Ricci curvature. E.g.: {(node1, node2): ricciCurvature}.

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
    global _cache_maxsize
    global _nbr_topk
    # -------------------------------------------------------

    _Gk = nk.nxadapter.nx2nk(G, weightAttr=weight)
    _alpha = alpha
    _weight = weight
    _method = method
    _base = base
    _exp_power = exp_power
    _proc = proc
    _cache_maxsize = cache_maxsize
    _nbr_topk = nbr_topk

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
    """Compute Ricci curvature of edges and nodes.
    The node Ricci curvature is defined as the average of node's adjacency edges.

    Parameters
    ----------
    G : NetworkX graph
        A given directional or undirectional NetworkX graph.
    weight : str
        The edge weight used to compute Ricci curvature. (Default value = "weight")
    **kwargs
        Additional keyword arguments passed to `_compute_ricci_curvature_edges`.

    Returns
    -------
    G: NetworkX graph
        A NetworkX graph with "ricciCurvature" on nodes and edges.
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
            logger.debug("node %s, Ricci Curvature = %f" % (n, G.nodes[n]['ricciCurvature']))

    return G


def _compute_ricci_flow(G: nx.Graph, weight="weight",
                        iterations=100, step=1, delta=1e-4, surgery=(lambda G, *args, **kwargs: G, 100),
                        **kwargs
                        ):
    """Compute the given Ricci flow metric of each edge of a given connected NetworkX graph.

    Parameters
    ----------
    G : NetworkX graph
        A given directional or undirectional NetworkX graph.
    weight : str
        The edge weight used to compute Ricci curvature. (Default value = "weight")
    iterations : int
        Iterations to require Ricci flow metric. (Default value = 100)
    step : float
        step size for gradient decent process. (Default value = 1)
    delta : float
        process stop when difference of Ricci curvature is within delta. (Default value = 1e-4)
    surgery : (function, int)
        A tuple of user define surgery function that will execute every certain iterations.
        (Default value = (lambda G, *args, **kwargs: G, 100))
    **kwargs
        Additional keyword arguments passed to `_compute_ricci_curvature`.

    Returns
    -------
    G: NetworkX graph
        A NetworkX graph with ``weight`` as Ricci flow metric.
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
    """A class to compute Ollivier-Ricci curvature for all nodes and edges in G.
    Node Ricci curvature is defined as the average of all it's adjacency edge.

    """

    def __init__(self, G: nx.Graph, weight="weight", alpha=0.5, method="OTD",
                 base=math.e, exp_power=2, proc=cpu_count(), chunksize=None, cache_maxsize=1000000,
                 nbr_topk=1000, verbose="ERROR"):
        """Initialized a container to compute Ollivier-Ricci curvature/flow.

        Parameters
        ----------
        G : NetworkX graph
            A given directional or undirectional NetworkX graph.
        weight : str
            The edge weight used to compute Ricci curvature. (Default value = "weight")
        edge_list : list of edges
            The list of edges to compute Ricci curvature, set to [] to run for all edges in G. (Default value = [])
        alpha : float
            The parameter for the discrete Ricci curvature, range from 0 ~ 1.
            It means the share of mass to leave on the original node.
            E.g. x -> y, alpha = 0.4 means 0.4 for x, 0.6 to evenly spread to x's nbr.
            (Default value = 0.5)
        method : {"OTD", "ATD", "Sinkhorn"}
            The optimal transportation distance computation method. (Default value = "OTD")

            Transportation method:
                - "OTD" for Optimal Transportation Distance,
                - "ATD" for Average Transportation Distance.
                - "Sinkhorn" for OTD approximated Sinkhorn distance.
        base : float
            Base variable for weight distribution. (Default value = `math.e`)
        exp_power : float
            Exponential power for weight distribution. (Default value = 0)
        proc : int
            Number of processor used for multiprocessing. (Default value = `cpu_count()`)
        chunksize : int
            Chunk size for multiprocessing, set None for auto decide. (Default value = `None`)
        cache_maxsize : int
            Max size for LRU cache for pairwise shortest path computation.
            Set this to `None` for unlimited cache. (Default value = 1000000)
        nbr_topk : int
            Only take the top k edge weight neighbors for density distribution.
            Smaller k run faster but the result is less accurate. (Default value = 1000)
        verbose: {"INFO","DEBUG","ERROR"}
            Verbose level. (Default value = "ERROR")
                - "INFO": show only iteration process log.
                - "DEBUG": show all output logs.
                - "ERROR": only show log if error happened.

        """
        self.G = G.copy()
        self.alpha = alpha
        self.weight = weight
        self.method = method
        self.base = base
        self.exp_power = exp_power
        self.proc = proc
        self.chunksize = chunksize
        self.cache_maxsize = cache_maxsize
        self.nbr_topk = nbr_topk

        self.set_verbose(verbose)
        self.lengths = {}  # all pair shortest path dictionary
        self.densities = {}  # density distribution dictionary

        assert importlib.util.find_spec("ot"), \
            "Package POT: Python Optimal Transport is required for Sinkhorn distance."

    def set_verbose(self, verbose):
        """Set the verbose level for this process.

        Parameters
        ----------
        verbose: {"INFO","DEBUG","ERROR"}
            Verbose level. (Default value = "ERROR")
                - "INFO": show only iteration process log.
                - "DEBUG": show all output logs.
                - "ERROR": only show log if error happened.

        """
        set_verbose(verbose)

    def compute_ricci_curvature_edges(self, edge_list=None):
        """Compute Ricci curvature for edges in  given edge lists.

        Parameters
        ----------
        edge_list : list of edges
            The list of edges to compute Ricci curvature, set to [] to run for all edges in G. (Default value = [])

        Returns
        -------
        output : dict[(int,int), float]
            A dictionary of edge Ricci curvature. E.g.: {(node1, node2): ricciCurvature}.
        """
        return _compute_ricci_curvature_edges(G=self.G, weight=self.weight, edge_list=edge_list,
                                              alpha=self.alpha, method=self.method,
                                              base=self.base, exp_power=self.exp_power,
                                              proc=self.proc, chunksize=self.chunksize,
                                              cache_maxsize=self.cache_maxsize, nbr_topk=self.nbr_topk)

    def compute_ricci_curvature(self):
        """Compute Ricci curvature of edges and nodes.
        The node Ricci curvature is defined as the average of node's adjacency edges.

        Returns
        -------
        G: NetworkX graph
            A NetworkX graph with "ricciCurvature" on nodes and edges.

        Examples
        --------
        To compute the Ollivier-Ricci curvature for karate club graph::

            >>> G = nx.karate_club_graph()
            >>> orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
            >>> orc.compute_ricci_curvature()
            >>> orc.G[0][1]
            {'weight': 1.0, 'ricciCurvature': 0.11111111071683011}
        """

        self.G = _compute_ricci_curvature(G=self.G, weight=self.weight,
                                          alpha=self.alpha, method=self.method,
                                          base=self.base, exp_power=self.exp_power,
                                          proc=self.proc, chunksize=self.chunksize, cache_maxsize=self.cache_maxsize,
                                          nbr_topk=self.nbr_topk)
        return self.G

    def compute_ricci_flow(self, iterations=10, step=1, delta=1e-4, surgery=(lambda G, *args, **kwargs: G, 100)):
        """Compute the given Ricci flow metric of each edge of a given connected NetworkX graph.

        Parameters
        ----------
        iterations : int
            Iterations to require Ricci flow metric. (Default value = 100)
        step : float
            step size for gradient decent process. (Default value = 1)
        delta : float
            process stop when difference of Ricci curvature is within delta. (Default value = 1e-4)
        surgery : (function, int)
            A tuple of user define surgery function that will execute every certain iterations.
            (Default value = (lambda G, *args, **kwargs: G, 100))

        Returns
        -------
        G: NetworkX graph
            A NetworkX graph with ``weight`` as Ricci flow metric.

        Examples
        --------
        To compute the Ollivier-Ricci flow for karate club graph::

            >>> G = nx.karate_club_graph()
            >>> orc_OTD = OllivierRicci(G, alpha=0.5, method="OTD", verbose="INFO")
            >>> orc_OTD.compute_ricci_flow(iterations=10)
            >>> orc_OTD.G[0][1]
            {'weight': 0.06399135316908759,
             'ricciCurvature': 0.18608249978652802,
             'original_RC': 0.11111111071683011}
        """
        self.G = _compute_ricci_flow(G=self.G, weight=self.weight,
                                     iterations=iterations, step=step, delta=delta, surgery=surgery,
                                     alpha=self.alpha, method=self.method,
                                     base=self.base, exp_power=self.exp_power,
                                     proc=self.proc, chunksize=self.chunksize, cache_maxsize=self.cache_maxsize,
                                     nbr_topk=self.nbr_topk)
        return self.G
