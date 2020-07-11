import networkx as nx
import numpy as np
import importlib


def ARI(G, cc, clustering_label="club"):
    """
    Computer the Adjust Rand Index (clustering accuracy) of clustering "cc" with clustering_label as ground truth.

    Parameters
    ----------
    G : NetworkX graph
        A given NetworkX graph with node attribute "clustering_label" as ground truth.
    cc : dict or list or list of set
        Predicted community.
    clustering_label : str
        Node attribute name for ground truth.

    Returns
    -------
    ari : float
        Adjust Rand Index for predicted community.
    """

    if importlib.util.find_spec("sklearn") is not None:
        from sklearn import preprocessing, metrics
    else:
        print("scikit-learn not installed...")
        return -1

    complexlist = nx.get_node_attributes(G, clustering_label)

    le = preprocessing.LabelEncoder()
    y_true = le.fit_transform(list(complexlist.values()))

    if isinstance(cc, dict):
        # python-louvain partition format
        y_pred = np.array([cc[v] for v in complexlist.keys()])
    elif isinstance(cc, list):
        # sklearn partition format
        y_pred = cc
    elif isinstance(cc[0], set):
        # networkx partition format
        predict_dict = {c: idx for idx, comp in enumerate(cc) for c in comp}
        y_pred = np.array([predict_dict[v] for v in complexlist.keys()])
    else:
        return -1

    return metrics.adjusted_rand_score(y_true, y_pred)


def my_surgery(G_origin: nx.Graph(), weight="weight", cut=0):
    """A simple surgery function that remove the edges with weight above a threshold

    Parameters
    ----------
    G : NetworkX graph
        A graph with ``weight`` as Ricci flow metric to cut.
    weight:
        The edge weight used as Ricci flow metric. (Default value = "weight")
    cut:
        Manually assigned cutoff point.

    Returns
    -------
    G : NetworkX graph
        A graph after surgery.
    """
    G = G_origin.copy()
    w = nx.get_edge_attributes(G, weight)

    assert cut >= 0, "Cut value should be greater than 0."
    if not cut:
        cut = (max(w.values()) - 1.0) * 0.6 + 1.0  # Guess a cut point as default

    to_cut = []
    for n1, n2 in G.edges():
        if G[n1][n2][weight] > cut:
            to_cut.append((n1, n2))
    print("*************** Surgery time ****************")
    print("* Cut %d edges." % len(to_cut))
    G.remove_edges_from(to_cut)
    print("* Number of nodes now: %d" % G.number_of_nodes())
    print("* Number of edges now: %d" % G.number_of_edges())
    cc = list(nx.connected_components(G))
    print("* Modularity now: %f " % nx.algorithms.community.quality.modularity(G, cc))

    print("* ARI now: %f " % ARI(G, cc))
    print("*********************************************")

    return G