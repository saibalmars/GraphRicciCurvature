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
from multiprocessing import Pool, cpu_count

import cvxpy as cvx
import networkx as nx
import numpy as np
import math
import ot

EPSILON = 1e-7


class OllivierRicci:

    def __init__(self, G, alpha=0.5, weight=None, method="OTD",
                 base=math.e, exp_power=0, proc=cpu_count(), verbose=False):
        """
        A class to compute Ollivier-Ricci curvature for all nodes and edges in G.
        Node Ricci curvature is defined as the average of all it's adjacency edge.
        :param G: A NetworkX graph.
        :param alpha: The parameter for the discrete Ricci curvature, range from 0 ~ 1.
                     It means the share of mass to leave on the original node.
                     eg. x -> y, alpha = 0.4 means 0.4 for x, 0.6 to evenly spread to x's nbr.
        :param weight: The edge weight used to compute Ricci curvature.
        :param method: Transportation method, "OTD" for Optimal Transportation Distance,
                                              "ATD" for Average Transportation Distance.
                                              "Sinkhorn" for OTD approximated Sinkhorn distance.
        :param base: Base variable for weight distribution.
        :param exp_power: Exponential power for weight distribution.
        :param verbose: Set True to output the detailed log.
        """
        self.G = G
        self.alpha = alpha
        self.weight = weight
        self.method = method
        self.base = base
        self.exp_power = exp_power
        self.proc = proc
        self.verbose = verbose
        self.EPSILON = EPSILON  # to prevent divided by zero

        self.lengths = {}  # all pair shortest path dictionary
        self.densities = {}  # density distribution dictionary

    def _get_all_pairs_shortest_path(self):
        """
        Pre-compute the all pair shortest paths of the assigned graph self.G
        """
        print("Start to compute all pair shortest path.")
        # Construct the all pair shortest path lookup
        if importlib.util.find_spec("networkit") is not None:
            import networkit as nk
            t0 = time.time()
            Gk = nk.nxadapter.nx2nk(self.G, weightAttr=self.weight)
            apsp = nk.distance.APSP(Gk).run().getDistances()
            lengths = {}
            for i, n1 in enumerate(self.G.nodes()):
                lengths[n1] = {}
                for j, n2 in enumerate(self.G.nodes()):
                    if apsp[i][j] < 1e300:  # to drop unreachable node
                        lengths[n1][n2] = apsp[i][j]
            print(time.time() - t0, " sec for all pair by NetworKit.")
            return lengths
        else:
            print("NetworKit not found, use NetworkX for all pair shortest path instead.")
            t0 = time.time()
            lengths = dict(nx.all_pairs_dijkstra_path_length(self.G, weight=self.weight))
            print(time.time() - t0, " sec for all pair.")
            return lengths

    def _get_edge_density_distributions(self):
        """
        Pre-compute densities distribution for all edges.
        """

        print("Start to compute all pair density distribution for directed graph.")
        densities = dict()

        t0 = time.time()

        # Construct the density distributions on each node
        def get_single_node_neighbor_distributions(neighbors, direction="successors"):

            # Get sum of distributions from x's all neighbors
            if direction == "predecessors":
                nbr_edge_weight_sum = sum([self.base ** (-(self.lengths[nbr][x]) ** self.exp_power)
                                           for nbr in neighbors])
            else:
                nbr_edge_weight_sum = sum([self.base ** (-(self.lengths[x][nbr]) ** self.exp_power)
                                           for nbr in neighbors])

            if nbr_edge_weight_sum > self.EPSILON:
                if direction == "predecessors":
                    result = [(1.0 - self.alpha) * (self.base ** (-(self.lengths[nbr][x]) ** self.exp_power)) /
                              nbr_edge_weight_sum for nbr in neighbors]
                else:
                    result = [(1.0 - self.alpha) * (self.base ** (-(self.lengths[x][nbr]) ** self.exp_power)) /
                              nbr_edge_weight_sum for nbr in neighbors]
            elif len(neighbors) == 0:
                return []
            else:
                print("Neighbor weight sum too small, list:", neighbors)
                result = [(1.0 - self.alpha) / len(neighbors)] * len(neighbors)
            result.append(self.alpha)
            return result

        if self.G.is_directed():
            for x in self.G.nodes():
                predecessors = get_single_node_neighbor_distributions(list(self.G.predecessors(x)), "predecessors")
                successors = get_single_node_neighbor_distributions(list(self.G.successors(x)))
                densities[x] = {"predecessors": predecessors, "successors": successors}
        else:
            for x in self.G.nodes():
                densities[x] = get_single_node_neighbor_distributions(list(self.G.neighbors(x)))

        print(time.time() - t0, " sec for edge density distribution construction")
        return densities

    def _distribute_densities(self, source, target):
        """
        Get the density distributions of source and target node, and the cost (all pair shortest paths) between
        all source's and target's neighbors.
        :param source: Source node
        :param target: Target node
        :return: (source's neighbors distributions, target's neighbors distributions, cost dictionary)
        """

        # Append source and target node into weight distribution matrix x,y
        source_nbr = list(self.G.predecessors(source)) if self.G.is_directed() else list(self.G.neighbors(source))
        target_nbr = list(self.G.successors(target)) if self.G.is_directed() else list(self.G.neighbors(target))

        # Distribute densities for source and source's neighbors as x
        if not source_nbr:
            source_nbr.append(source)
            x = [1]
        else:
            source_nbr.append(source)
            x = self.densities[source]["predecessors"] if self.G.is_directed() else self.densities[source]

        # Distribute densities for target and target's neighbors as y
        if not target_nbr:
            target_nbr.append(target)
            y = [1]
        else:
            target_nbr.append(target)
            y = self.densities[target]["successors"] if self.G.is_directed() else self.densities[target]

        # construct the cost dictionary from x to y
        d = np.zeros((len(x), len(y)))

        for i, src in enumerate(source_nbr):
            for j, dst in enumerate(target_nbr):
                assert dst in self.lengths[src], \
                    "Target node not in list, should not happened, pair (%d, %d)" % (src, dst)
                d[i][j] = self.lengths[src][dst]

        x = np.array([x]).T  # the mass that source neighborhood initially owned
        y = np.array([y]).T  # the mass that target neighborhood needs to received

        return x, y, d

    def _optimal_transportation_distance(self, x, y, d):
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
        if self.verbose:
            print(time.time() - t0, " secs for cvxpy.")
            print("\t#source_nbr: %d, #target_nbr: %d, " % (len(x), len(y)), end='')
        return m

    def _average_transportation_distance(self, source, target):
        """
        Compute the average transportation distance (ATD) of the given density distributions.
        :param source: Source node
        :param target: Target node
        :return: Average transportation distance
        """

        t0 = time.time()
        source_nbr = list(self.G.predecessors(source)) if self.G.is_directed() else list(self.G.neighbors(source))
        target_nbr = list(self.G.successors(target)) if self.G.is_directed() else list(self.G.neighbors(target))

        share = (1.0 - self.alpha) / (len(source_nbr) * len(target_nbr))
        cost_nbr = 0
        cost_self = self.alpha * self.lengths[source][target]

        for i, s in enumerate(source_nbr):
            for j, t in enumerate(target_nbr):
                assert t in self.lengths[s], "Target node not in list, should not happened, pair (%d, %d)" % (s, t)
                cost_nbr += self.lengths[s][t] * share

        m = cost_nbr + cost_self  # Average transportation cost

        if self.verbose:
            print("%8f secs for Average Transportation Distance." % (time.time() - t0))
            print("\t#source_nbr: %d, #target_nbr: %d, " % (len(source_nbr), len(target_nbr)), end='')
        return m

    def _compute_ricci_curvature_single_edge(self, source, target):
        """
        Ricci curvature computation process for a given single edge.

        :param source: The source node
        :param target: The target node
        :return: The Ricci curvature of given edge

        """

        assert source != target, "Self loop is not allowed."  # to prevent self loop

        # If the weight of edge is too small, return 0 instead.
        if self.lengths[source][target] < self.EPSILON:
            print("Zero Weight edge detected, return Ricci Curvature as 0 instead.")
            return {(source, target): 0}

        # compute transportation distance
        m = 1  # assign an initial cost
        assert self.method in ["OTD", "ATD", "Sinkhorn"], \
            'Method %s not found, support method:["OTD", "ATD", "Sinkhorn"]' % self.method
        if self.method == "OTD":
            x, y, d = self._distribute_densities(source, target)
            m = self._optimal_transportation_distance(x, y, d)
        elif self.method == "ATD":
            m = self._average_transportation_distance(source, target)
        elif self.method == "Sinkhorn":
            assert importlib.util.find_spec("ot"), \
                "Package POT: Python Optimal Transport is required for Sinkhorn distance."
            x, y, d = self._distribute_densities(source, target)
            m = ot.sinkhorn2(x, y, d, 1e-1, method='sinkhorn')[0]

        # compute Ricci curvature: k=1-(m_{x,y})/d(x,y)
        result = 1 - (m / self.lengths[source][target])  # Divided by the length of d(i, j)
        if self.verbose:
            print("Ricci curvature = %f" % result)
        return {(source, target): result}

    def _wrap_compute_single_edge(self, stuff):
        """
        Wrapper for args in multiprocessing
        """
        return self._compute_ricci_curvature_single_edge(*stuff)

    def compute_ricci_curvature_edges(self, edge_list=None):
        """
        Compute Ricci curvature of given edge lists.

        :return: G: A NetworkX graph with Ricci Curvature with edge attribute "ricciCurvature"
        """

        if not edge_list:
            edge_list = []

        # Construct the all pair shortest path dictionary
        if not self.lengths:
            self.lengths = self._get_all_pairs_shortest_path()

        # Construct the density distribution
        if not self.densities:
            self.densities = self._get_edge_density_distributions()

        # Start compute edge Ricci curvature
        t0 = time.time()

        p = Pool(processes=self.proc)

        # Compute Ricci curvature for edges
        args = [(source, target) for source, target in edge_list]

        result = p.map_async(self._wrap_compute_single_edge, args)
        result = result.get()
        p.close()
        p.join()
        print(time.time() - t0, " sec for Ricci curvature computation.")

        return result

    def compute_ricci_curvature(self):
        """
        Compute Ricci curvature of edges and nodes.
        The node Ricci curavture is defined as the average of node's adjacency edges.

        :return: G: A NetworkX graph with Ricci Curvature with edge attribute "ricciCurvature"
        """

        edge_ricci = self.compute_ricci_curvature_edges(self.G.edges())

        # Assign edge Ricci curvature from result to graph G
        for rc in edge_ricci:
            for k in list(rc.keys()):
                source, target = k
                self.G[source][target]['ricciCurvature'] = rc[k]

        # Compute node Ricci curvature
        for n in self.G.nodes():
            rc_sum = 0  # sum of the neighbor Ricci curvature
            if self.G.degree(n) != 0:
                for nbr in self.G.neighbors(n):
                    if 'ricciCurvature' in self.G[n][nbr]:
                        rc_sum += self.G[n][nbr]['ricciCurvature']

                # Assign the node Ricci curvature to be the average of node's adjacency edges
                self.G.node[n]['ricciCurvature'] = rc_sum / self.G.degree(n)
                if self.verbose:
                    print("node %d, Ricci Curvature = %f" % (n, self.G.node[n]['ricciCurvature']))

        return self.G

    def compute_ricci_flow(self, iterations=100, step=1, delta=1e-4, surgery=(lambda G, *args, **kwargs: G, 100)):
        """
        Compute the given Ricci flow metric of each edge of a given connected Networkx graph.

        :param iterations: Iterations to require Ricci flow metric
        :param step: step size for gradient decent process
        :param delta: process stop when difference of Ricci curvature is within delta
        :param surgery: A tuple of user define surgery function that will execute every certain iterations.
        :return: A network graph G with "weight" as Ricci flow metric
        """

        if not nx.is_connected(self.G):
            print("Not connected graph detected, compute on the largest connected component instead.")
            self.G = nx.Graph(max(nx.connected_component_subgraphs(self.G), key=len))
        self.G.remove_edges_from(self.G.selfloop_edges())

        print("Number of nodes: %d" % self.G.number_of_nodes())
        print("Number of edges: %d" % self.G.number_of_edges())

        # Set normalized weight to be the number of edges.
        normalized_weight = float(self.G.number_of_edges())

        if not self.weight:
            print("No specific weight detected, use \"weight\" as weight")
            self.weight = "weight"
            weight = "weight"
        else:
            weight = self.weight

        if nx.get_edge_attributes(self.G, "original_RC"):
            print("original_RC detected, continue to refine the ricci flow.")
        else:
            if not nx.get_edge_attributes(self.G, weight):
                for (v1, v2) in self.G.edges():
                    self.G[v1][v2][weight] = 1.0
            else:
                for (v1, v2) in self.G.edges():
                    self.G[v1][v2]["original_%s" % weight] = self.G[v1][v2][weight]
                    self.G[v1][v2][weight] = 1.0

            self.G = self.compute_ricci_curvature()

            for (v1, v2) in self.G.edges():
                self.G[v1][v2]["original_RC"] = self.G[v1][v2]["ricciCurvature"]

        # Start the Ricci flow process
        for i in range(iterations):
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][weight] -= step * (self.G[v1][v2]["ricciCurvature"]) * self.G[v1][v2][weight]

            # Do normalization on all weight to prevent weight expand to infinity
            w = nx.get_edge_attributes(self.G, weight)
            sumw = sum(w.values())
            for k, v in w.items():
                w[k] = w[k] * (normalized_weight / sumw)
            nx.set_edge_attributes(self.G, w, weight)
            print("= Ricci flow iteration %d =" % i)

            self.compute_ricci_curvature()

            rc = nx.get_edge_attributes(self.G, "ricciCurvature")
            diff = max(rc.values()) - min(rc.values())
            print("Ricci curvature difference:", diff)
            print("max:%f, min:%f|||| maxw:%f, minw:%f" % (
                max(rc.values()), min(rc.values()), max(w.values()), min(w.values())))

            if diff < delta:
                print("Ricci Curvature converged, process terminated.")
                break

            # do surgery or any specific evaluation
            surgery_func, do_surgery = surgery
            if i != 0 and i % do_surgery == 0:
                self.G = surgery_func(self.G, weight)
                normalized_weight = float(self.G.number_of_edges())

            if self.verbose:
                for n1, n2 in self.G.edges():
                    print(n1, n2, self.G[n1][n2])

            # clear the APSP and densities since the graph have changed.
            self.lengths = {}
            self.densities = {}

        return self.G
