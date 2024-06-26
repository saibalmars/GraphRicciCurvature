import logging
import community.community_louvain as community_louvain
import networkx as nx
import numpy as np
from functools import partial, partialmethod

logging.TRACE = logging.DEBUG + 5
logging.addLevelName(logging.TRACE, 'TRACE')
logging.Logger.trace = partialmethod(logging.Logger.log, logging.TRACE)
logging.trace = partial(logging.log, logging.TRACE)

logger = logging.getLogger("GraphRicciCurvature")


def set_verbose(verbose="ERROR"):
    """Set up the verbose level of the GraphRicciCurvature.

    Parameters
    ----------
    verbose : {"INFO", "TRACE","DEBUG","ERROR"}
        Verbose level. (Default value = "ERROR")
            - "INFO": show only iteration process log.
            - "TRACE": show detailed iteration process log.
            - "DEBUG": show all output logs.
            - "ERROR": only show log if error happened.
    """
    if verbose == "INFO":
        logger.setLevel(logging.INFO)
    elif verbose == "TRACE":
        logger.setLevel(logging.TRACE)
    elif verbose == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif verbose == "ERROR":
        logger.setLevel(logging.ERROR)
    else:
        print('Incorrect verbose level, option:["INFO","DEBUG","ERROR"], use "ERROR instead."')
        logger.setLevel(logging.ERROR)


def cut_graph_by_cutoff(G_origin, cutoff, weight="weight"):
    """Remove graph's edges with "weight" greater than "cutoff".

    Parameters
    ----------
    G_origin : NetworkX graph
        A graph with ``weight`` as Ricci flow metric to cut.
    cutoff : float
        A threshold to remove all edges with "weight" greater than it.
    weight : str
        The edge weight used as Ricci flow metric. (Default value = "weight")
    Returns
    -------

    G: NetworkX graph
        A graph with edges cut by given cutoff value.
    """
    assert nx.get_edge_attributes(G_origin, weight), "No edge weight detected, abort."

    G = G_origin.copy()
    edge_trim_list = []
    for n1, n2 in G.edges():
        if G[n1][n2][weight] > cutoff:
            edge_trim_list.append((n1, n2))
    G.remove_edges_from(edge_trim_list)
    return G


def get_rf_metric_cutoff(G_origin, weight="weight", cutoff_step=0.025, drop_threshold=0.01):
    """Get good clustering cutoff points for Ricci flow metric by detect the change of modularity while removing edges.

    Parameters
    ----------
    G_origin : NetworkX graph
        A graph with "weight" as Ricci flow metric to cut.
    weight : str
        The edge weight used as Ricci flow metric. (Default value = "weight")
    cutoff_step : float
        The step size to find the good cutoff points.
    drop_threshold : float
        At least drop this much to considered as a drop for good_cut.

    Returns
    -------
    good_cuts : list of float
        A list of possible cutoff point, usually we use the first one as the best cut.
    """

    G = G_origin.copy()
    modularity, ari = [], []
    maxw = max(nx.get_edge_attributes(G, weight).values())
    cutoff_range = np.arange(maxw, 1, -cutoff_step)

    for cutoff in cutoff_range:
        G = cut_graph_by_cutoff(G, cutoff, weight=weight)
        # Get connected component after cut as clustering
        clustering = {c: idx for idx, comp in enumerate(nx.connected_components(G)) for c in comp}
        # Compute modularity
        modularity.append(community_louvain.modularity(clustering, G, weight))

    good_cuts = []
    mod_last = modularity[-1]

    # check drop from 1 -> maxw
    for i in range(len(modularity) - 1, 0, -1):
        mod_now = modularity[i]
        if mod_last > mod_now > 1e-4 and abs(mod_last - mod_now) / mod_last > drop_threshold:
            logger.trace("Cut detected: cut:%f, diff:%f, mod_now:%f, mod_last:%f" % (
                cutoff_range[i+1], mod_last - mod_now, mod_now, mod_last))
            good_cuts.append(cutoff_range[i+1])
        mod_last = mod_now

    return good_cuts
