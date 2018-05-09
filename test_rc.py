from __future__ import print_function

__author__ = 'saiba'
from multiprocessing import Pool
import time
import math
import networkx as nx
import cvxpy as cvx
import numpy as np
import ot
import random
import importlib

def _wrapRicci(stuff):
    """
    A wrapper for ricciCurvature_parallel for multiple args,
    used for trimmed single edge process
    """

    if _method == "base":
        return ricciCurvature_singleEdge_luo(*stuff)
    elif _method == "uniformly":
        return ricciCurvature_singleEdge_uniformly(*stuff)
    elif _method == "sinkhorn":
        return ricciCurvature_singleEdge_ot_sinkhorn(*stuff)
    elif _method == "mix":
        if random.random() > 0.5:
            return ricciCurvature_singleEdge_luo(*stuff)
        else:
            return ricciCurvature_singleEdge_ot_sinkhorn(*stuff)


def ricciCurvature_singleEdge_ot_sinkhorn(source, target):
    """
        By POT optimal transportation library
        Ricci curvature computation process for a given single edge, use e as base to distribute weight

        Notice: global variables are required.

        input:
            source: the index of the source node
            target: the index of the target node
        output:
            result: a dict that store the (source, target) tuple as key,
                    and the corresponding ricci curvature as value.
        required:
            _G:     the original graph
            _alpha: ricci curvature parameter
            _length: all pair shortest paths dict

    """

    result = dict()
    EPSILON = 1e-6  # to prevent divided by zero

    # For ricci flow, to prevent the weight of edge to be too small,
    # return the previous ricci Curvature instead.
    if _length[source][target] < EPSILON:
        assert "ricciCurvature" in _G[source][target], "Divided by Zero and no ricci curvature exist in Graph!"
        if not _silence:
            print("Zero Weight edge detected, return nothing instead.")
        # result[source, target]=0
        return result

    sourceNbr = list(_G.neighbors(source)) + [source]
    targetNbr = list(_G.neighbors(target)) + [target]
    x = _densityDist[source]
    y = _densityDist[target]

    # construct the cost dictionary from x to y
    d = np.zeros((len(x), len(y)))

    for i, s in enumerate(sourceNbr):
        for j, t in enumerate(targetNbr):
            d[i][j] = _length[s][t]

    t0 = time.time()

    if not _silence:
        print("solving for sinkhorn", source, target, end=' ')

    result[source, target] = (ot.sinkhorn2(x, y, d, 1e-1, method='sinkhorn')[0])
    if not _silence:
        print(time.time() - t0, " secs for sinkhorn.", end=' ')
        print(len(x), len(y), result[source, target])

    return result


def ricciCurvature_singleEdge_luo(source, target):
    """
        Ricci curvature computation process for a given single edge, use e as base to distribute weight

        Notice: global variables are required.

        input:
            source: the index of the source node
            target: the index of the target node
        output:
            result: a dict that store the (source, target) tuple as key,
                    and the corresponding ricci curvature as value.
        required:
            _G:     the original graph
            _alpha: ricci curvature parameter
            _length: all pair shortest paths dict

    """

    result = dict()
    EPSILON = 1e-6  # to prevent divided by zero

    # For ricci flow, to prevent the weight of edge to be too small,
    # return the previous ricci Curvature instead.
    if _length[source][target] < EPSILON:
        assert "ricciCurvature" in _G[source][target], "Divided by Zero and no ricci curvature exist in Graph!"
        if not _silence:
            print("Zero Weight edge detected, return 0 instead.")
        result[source, target]=0
        return result

    sourceNbr = list(_G.neighbors(source)) + [source]
    targetNbr = list(_G.neighbors(target)) + [target]
    x = _densityDist[source]
    y = _densityDist[target]

    # construct the cost dictionary from x to y
    d = np.zeros((len(x), len(y)))

    for i, s in enumerate(sourceNbr):
        for j, t in enumerate(targetNbr):
            d[i][j] = _length[s][t]

    t0 = time.time()

    x = np.array([x]).T  # the mass that source neighborhood initially owned
    y = np.array([y]).T  # the mass that target neighborhood needs to received

    rho = cvx.Variable(len(y), len(x))  # the transportation plan rho

    # objective function d(x,y) * rho * x, need to do element-wise multiply here
    obj = cvx.Minimize(cvx.sum_entries(cvx.mul_elemwise(np.multiply(d.T, x.T), rho)))

    # \sigma_i rho_{ij}=[1,1,...,1]
    source_sum = cvx.sum_entries(rho, axis=0)
    constrains = [rho * x == y, source_sum == np.ones((1, (len(x)))), 0 <= rho, rho <= 1]
    prob = cvx.Problem(obj, constrains)

    if not _silence:
        print("solving for ", source, target, end=' ')
    try:
        prob.solve(solver="ECOS_BB")
        # prob.solve(solver="SCS")
    except:
        print("STATUS: %s, Failed to solve it, return nothing instead"%prob.status)
        return result
    else:
        if prob.status == "optimal" or prob.status == "optimal_inaccurate":  # solve success
            result[source, target] = prob.value
        else:
            print("Solver fail: %s, solver value: %f" % (prob.status, prob.value))
            print("(source,target)=", source, target)
            return result

    assert prob.status == "optimal" or prob.status == "optimal_inaccurate"

    if not _silence:
        print(time.time() - t0, " secs for cvxpy.", end=' ')
        print(len(x), len(y), result[source, target])

    return result


def ricciCurvature_singleEdge_uniformly(source, target):
    """
        Ricci curvature computation process for a given single edge.
        By the uniform distribution way.

        Notice: global variables are required.

        input:
            source: the index of the source node
            target: the index of the target node
        output:
            result: a dict that store the (source, target) tuple as key,
                    and the corresponding ricci curvature as value.
        required:
            _G:     the original graph
            _alpha: ricci curvature parameter
            _length: all pair shortest paths dict

    """

    result = {}  # the final result dictionary
    EPSILON = 1e-6  # to prevent divided by zero

    # For ricci flow, to prevent the weight of edge to be too small,
    # return the previous ricci Curvature instead.
    if _length[source][target] < EPSILON:
        assert "ricciCurvature" in _G[source][target], "Divided by Zero and no ricci curvature exist in Graph!"
        result[source, target] = _G[source][target]["ricciCurvature"]
        if not _silence:
            print("Zero Weight edge detected, return average ricci Curvature instead.")
        return result

    t0 = time.time()
    sourceNbr = list(_G.neighbors(source))
    targetNbr = list(_G.neighbors(target))

    share = (1.0 - _alpha) / (len(sourceNbr) * len(targetNbr))
    costNbr = 0
    costSelf = _alpha * _length[source][target]

    for i, s in enumerate(sourceNbr):
        for j, t in enumerate(targetNbr):
            assert t in _length[s], "Target node not in list, should not happened, pair (%d, %d)" % (s, t)
            costNbr += _length[s][t] * share

    result[source, target] = costNbr + costSelf

    if not _silence:
        print(time.time() - t0, " secs for uniform.", end=' ')
        print(len(sourceNbr), len(targetNbr), result[source, target])

    return result


def _ricciParallel(proc, args):
    """
    The main parallel part of the ricci curvature computation.

    :rtype : dict
    :param proc: Number of processor to used.
    :return: A result ricci curvature dictionary
    """
    p = Pool(processes=proc)
    result = p.map_async(_wrapRicci, args)
    result = result.get()
    p.close()
    p.join()
    return result


def ricciCurvature_parallel(G, alpha=0.5, proc=8, weight=None, method="base", base=math.e, exp_power=1, silence=True):
    """
        Compute ricci curvature for all nodes and edges in G by multiprocessing.
        Node ricci curvature is defined as the average of all it's adjacency edge.

        input:
            G:      a networkx graph.
            alpha:  the parameter for the discrete ricci curvature, range from 0 ~ 1.
                    It means the share of mass to leave on the original node.
                    eg. x -> y, alpha = 0.4 means 0.4 for x, 0.6 to evenly spread to x's nbr.
            proc:   Number of processor used.
            weight: weight to use to compute ricci curvature
            method: the method to compute ricci curvature["base","sinkhorn","mix","uniformly"]
            base:   the weight distribution base, default is math.e
            exp_power: the distribution power of the base, deafult is 1

    """

    # ---set to global variable for multiprocessing used.---
    global _G
    global _method
    global _silence
    global _length
    global _densityDist
    global _alpha
    # -------------------------------------------------------

    _G = G
    _silence = silence
    _method = method
    _alpha = alpha

    eps=1e-6

    # Construct the all pair shortest path lookup

    # Construct the all pair shortest path lookup
    if importlib.util.find_spec("networkit") is not None:
        import networkit as nk
        t0 = time.time()
        Gk = nk.nxadapter.nx2nk(G, weightAttr=weight)
        apsp = nk.distance.APSP(Gk).run().getDistances()
        _length = {}
        for i, n1 in enumerate(G.nodes()):
            _length[n1] = {}
            for j, n2 in enumerate(G.nodes()):
                _length[n1][n2] = apsp[i][j]
        print(time.time() - t0, " sec for all pair by NetworKit.")
    else:
        print("NetworKit not found, use NetworkX for all pair shortest path instead.")
        t0 = time.time()
        _length = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))

        print(time.time() - t0, " sec for all pair.")

    _densityDist = dict()

    t0 = time.time()

    # Construct the density distribution on each node
    for x in G.nodes():
        St_x = list(G.neighbors(x))
        nbr_edge_weight_sum = sum([base ** (-(_length[x][nbr]) ** exp_power) for nbr in St_x])
        if nbr_edge_weight_sum > eps:
            _densityDist[x] = [(1.0 - alpha) * (base ** (-(_length[x][nbr]) ** exp_power)) / nbr_edge_weight_sum for nbr in
                           St_x]
        elif len(St_x)==0:
            continue
        else:
            print("nbr weight sum too small, list:",St_x)
            _densityDist[x] = [(1.0 - alpha) / len(St_x)] * len(St_x)
        _densityDist[x].append(alpha)

    print(time.time() - t0, " sec for density distribution construction")

    # construct arguments for parallel distribution
    args = [(source, target) for source, target in G.edges()]

    # compute ricci curvature
    t0 = time.time()
    result = _ricciParallel(proc, args)
    print()

    # assign edge ricci curvature from result to graph G
    for j in result:
        for k in list(j.keys()):
            source, target = k
            if _length[source][target] >eps:   # update the ricci curvature only if _length[source][target] not 0
                G[source][target]['ricciCurvature'] = 1.0 - float(j[k]) / _length[source][target]
            else:
                G[source][target]['ricciCurvature'] = 0
    if not _silence:
        print("Edge ricci curvature computation done.")
    print(time.time() - t0, " sec for parallel edge ricci computation")
    # print()

    # compute node ricci curvature to graph G
    for n in G.nodes():
        rcsum = 0  # sum of the neighbor ricci curvature
        if G.degree(n) != 0:
            for nbr in G.neighbors(n):
                if 'ricciCurvature' in G[n][nbr]:
                    rcsum += G[n][nbr]['ricciCurvature']

            # assign the node ricci curvature to be the average of node's adjacency edges
            G.node[n]['ricciCurvature'] = rcsum / G.degree(n)

            if not _silence:
                print("node", n, G.degree(n), G.node[n]['ricciCurvature'])
    if not _silence:
        print("Node ricci curvature computation done.")
    print()


    if not _silence:
        for n1, n2 in G.edges():
            print(n1, n2, G[n1][n2])
    return G


if __name__ == '__main__':

    # G=nx.read_gexf("../data/polblogs.gexf")
    # G=nx.karate_club_graph()
    G=nx.random_regular_graph(8,1000)

    # G.remove_edges_from(nx.selfloop_edges(G))

    for (v1, v2) in G.edges():
        G[v1][v2]["weight"] = 1.0

    #
    # Grf=ricciCurvature_parallel(G, proc=8, alpha=0)
    G = ricciCurvature_parallel(G, proc=8, alpha=0.5, weight="weight", method="sinkhorn")


