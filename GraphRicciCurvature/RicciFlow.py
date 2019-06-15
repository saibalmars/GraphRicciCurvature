"""
A NetworkX addon program to compute the Ollivier-Ricci Flow of a given NetworkX graph.


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
[Ni2]: Ni, C.-C., Lin, Y.-Y., Gao, J., and Gu, X. . "Network Alignment by Discrete Ollivier-Ricci Flow", Graph Drawing 2018

"""

from .OllivierRicci import *


def compute_ricciFlow(G, iterations=100, alpha=0.5, eps=1, delta=1e-4, proc=8, method="OTD", verbose=False):
    """
    Compute the given Ricci flow metric of each edge of a given connected Networkx graph.

    :param G: A connected networkx graph
    :param iterations: Iterations to require ricci flow metric
    :param alpha: alpha value for Ricci curvature
    :param eps: step size for gradient decent process
    :param delta: process stop when difference of Ricci curvature is within delta
    :param proc: number of parallel processor for computation
    :param method: the method to compute Ricci curvature["OTD", "ATD"]
    :param verbose: print log or not
    :return: A network graph G with "weight" as Ricci flow metric
    """

    if not nx.is_connected(G):
        print("Not connected graph detected, compute on the largest connected component instead.")
        G = nx.Graph(max(nx.connected_component_subgraphs(G), key=len))
    G.remove_edges_from(G.selfloop_edges())

    print("Number of nodes: %d" % G.number_of_nodes())
    print("Number of edges: %d" % G.number_of_edges())

    normalized_weight = float(G.number_of_edges())

    if nx.get_edge_attributes(G, "original_RC"):
        print("original_RC detected, continue to refine the ricci flow.")
    else:
        if not nx.get_edge_attributes(G, "weight"):
            for (v1, v2) in G.edges():
                G[v1][v2]["weight"] = 1.0
        else:
            for (v1, v2) in G.edges():
                G[v1][v2]["original_weight"] = G[v1][v2]["weight"]
                G[v1][v2]["weight"] = 1.0

        G = ricciCurvature(G, alpha=alpha, proc=proc, weight="weight", method=method, verbose=verbose)
        for (v1, v2) in G.edges():
            G[v1][v2]["original_RC"] = G[v1][v2]["ricciCurvature"]

    for i in range(iterations):
        for (v1, v2) in G.edges():
            G[v1][v2]["weight"] -= eps * (G[v1][v2]["ricciCurvature"]) * G[v1][v2]["weight"]

        # do normalized on all weight
        w = nx.get_edge_attributes(G, "weight")
        sumw = sum(w.values())
        for k, v in w.items():
            w[k] = w[k] * (normalized_weight / sumw)
        nx.set_edge_attributes(G, w, "weight")

        print("Round" + str(i))
        G = ricciCurvature(G, alpha=alpha, proc=proc, weight="weight", method=method, verbose=verbose)

        rc = nx.get_edge_attributes(G, "ricciCurvature")

        diff = max(rc.values()) - min(rc.values())
        print("Ricci curvature difference:", diff)
        if diff < delta:
            print("Ricci Curvature converged, process terminated.")

        if verbose:
            for n1, n2 in G.edges():
                print(n1, n2, G[n1][n2])

    return G
