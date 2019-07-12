import networkx as nx


def my_surgery(G: nx.Graph(), weight="weight"):
    """
    A simple surgery function that remove the edges with weight above a threshold
    :param G: A weighted networkx graph.
    :param weight: name of edge weight to cut.
    :return: A weighted networkx graph after surgery.
    """

    w = nx.get_edge_attributes(G, weight)
    cut = (max(w.values()) - 1.0) * 0.6 + 1.0
    if cut > 1:
        to_cut = []
        for n1, n2 in G.edges():
            if G[n1][n2][weight] > cut:
                to_cut.append((n1, n2))
        print("*************** Surgery time ****************")
        print("* Cut %d edges." % len(to_cut))
        G.remove_edges_from(to_cut)
        print("* Number of nodes now: %d" % G.number_of_nodes())
        print("* Number of edges now: %d" % G.number_of_edges())
        print("*********************************************")
    return G