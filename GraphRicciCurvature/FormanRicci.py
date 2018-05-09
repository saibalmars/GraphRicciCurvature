"""
A NetworkX addon program to compute the Forman-Ricci curvature of a given NetworkX graph.



Author:
    Chien-Chun Ni
    http://www3.cs.stonybrook.edu/~chni/

Reference:
    Forman. 2003. “Bochner’s Method for Cell Complexes and Combinatorial Ricci Curvature.” Discrete & Computational Geometry 29 (3). Springer-Verlag: 323–74.
    Sreejith, R. P., Karthikeyan Mohanraj, Jürgen Jost, Emil Saucan, and Areejit Samal. 2016. “Forman Curvature for Complex Networks.” Journal of Statistical Mechanics: Theory and Experiment 2016 (6). IOP Publishing: 063206.

"""


def formanCurvature(G, verbose=False):
    """
     Compute Forman-ricci curvature for all nodes and edges in G.
         Node curvature is defined as the average of all it's adjacency edge.
     :param G: A connected NetworkX graph, unweighted graph only now, edge weight will be ignored.
     :param verbose: Show detailed logs.
     :return: G: A NetworkX graph with Forman-Ricci Curvature with node and edge attribute "formanCurvature"
     """

    # Edge forman curvature
    for (v1, v2) in G.edges():
        if G.is_directed():
            v1_nbr = set(list(G.predecessors(v1)) + list(G.successors(v1)))
            v2_nbr = set(list(G.predecessors(v2)) + list(G.successors(v2)))
        else:
            v1_nbr = set(G.neighbors(v1))
            v1_nbr.remove(v2)
            v2_nbr = set(G.neighbors(v2))
            v2_nbr.remove(v1)
        face = v1_nbr & v2_nbr
        # G[v1][v2]["face"]=face
        prl_nbr = (v1_nbr | v2_nbr) - face
        # G[v1][v2]["prl_nbr"]=prl_nbr

        G[v1][v2]["formanCurvature"] = len(face) + 2 - len(prl_nbr)
        if verbose:
            print("Source: %s, target: %d, Forman-Ricci curvature = %f  " % (v1, v2, G[v1][v2]["formanCurvature"]))

    # Node forman curvature
    for n in G.nodes():
        fcsum = 0  # sum of the neighbor Forman curvature
        if G.degree(n) != 0:
            for nbr in G.neighbors(n):
                if 'formanCurvature' in G[n][nbr]:
                    fcsum += G[n][nbr]['formanCurvature']

            # assign the node Forman curvature to be the average of node's adjacency edges
            G.node[n]['formanCurvature'] = fcsum / G.degree(n)
        if verbose:
            print("node %d, Forman Curvature = %f" % (n, G.node[n]['formanCurvature']))
    print("Forman curvature computation done.")

    return G
