import networkx as nx
import numpy.testing as npt

from GraphRicciCurvature.FormanRicci import FormanRicci


def test_compute_ricci_curvature():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (2, 4)])
    G.add_node(5)
    frc = FormanRicci(G, method="1d")
    frc.compute_ricci_curvature()

    frc_edges = list(nx.get_edge_attributes(frc.G, "formanCurvature").values())
    frc_nodes = list(nx.get_node_attributes(frc.G, "formanCurvature").values())
    frc_edges_ans = [0.0, -1.0, -1.0, 0.0]
    frc_nodes_ans = [0.0, -0.6666666666666666, -0.5, -0.5, 0]

    npt.assert_array_almost_equal(frc_edges, frc_edges_ans)
    npt.assert_array_almost_equal(frc_nodes, frc_nodes_ans)

    frc_a = FormanRicci(G, method="augmented")
    frc_a.compute_ricci_curvature()

    frc_a_edges = list(nx.get_edge_attributes(frc_a.G, "formanCurvature").values())
    frc_a_nodes = list(nx.get_node_attributes(frc_a.G, "formanCurvature").values())
    frc_a_edges_ans = [0.0, 2.0, 2.0, 3.0]
    frc_a_nodes_ans = [0.0, 1.3333333333333333, 2.5, 2.5, 0]

    npt.assert_array_almost_equal(frc_a_edges, frc_a_edges_ans)
    npt.assert_array_almost_equal(frc_a_nodes, frc_a_nodes_ans)

    # balanced Forman-Ricci curvature (Topping et al. 2022):
    # edge (1,2) is incident to a degree-1 node -> 0 by Definition 1;
    # edge (2,3) and (2,4): one triangle, no 4-cycle -> 5/6;
    # edge (3,4): one triangle, no 4-cycle -> 3/2.
    frc_b = FormanRicci(G, method="balanced")
    frc_b.compute_ricci_curvature()

    frc_b_edges = list(nx.get_edge_attributes(frc_b.G, "formanCurvature").values())
    frc_b_nodes = list(nx.get_node_attributes(frc_b.G, "formanCurvature").values())
    frc_b_edges_ans = [0.0, 0.8333333333333334, 0.8333333333333334, 1.5]
    frc_b_nodes_ans = [0.0, 0.5555555555555556, 1.1666666666666667, 1.1666666666666667, 0]

    npt.assert_array_almost_equal(frc_b_edges, frc_b_edges_ans)
    npt.assert_array_almost_equal(frc_b_nodes, frc_b_nodes_ans)

    # TODO: test for directed part

    # Gd = nx.DiGraph()
    # Gd.add_edges_from([(1, 2), (2, 3), (3, 4), (2, 4), (4, 2)])
    # Gd.add_node(5)
    # frc_directed = FormanRicci(Gd)
    # frc_directed.compute_ricci_curvature()
    #
    # frc = nx.get_edge_attributes(frc_directed.G, "formanCurvature")
    #
    # assert frc == {(1, 2): -2, (2, 3): 0, (2, 4): 0, (3, 4): 1, (4, 2): 0}

def test_compute_balanced_forman_curvature():
    # C4: degree 2, no triangle; the whole C4 is a 4-cycle based at each
    # edge with degeneracy 1 -> Ric = 2/2 + 2/2 - 2 + (1+1)/2 = 1.
    G = nx.cycle_graph(4)
    frc = FormanRicci(G, method="balanced")
    frc.compute_ricci_curvature()
    npt.assert_array_almost_equal(
        list(nx.get_edge_attributes(frc.G, "formanCurvature").values()), [1.0] * 4)
    npt.assert_array_almost_equal(
        list(nx.get_node_attributes(frc.G, "formanCurvature").values()), [1.0] * 4)

    # K3: one triangle per edge, degree 2 -> 3/2.
    G = nx.complete_graph(3)
    frc = FormanRicci(G, method="balanced")
    frc.compute_ricci_curvature()
    npt.assert_array_almost_equal(
        list(nx.get_edge_attributes(frc.G, "formanCurvature").values()), [1.5] * 3)
    npt.assert_array_almost_equal(
        list(nx.get_node_attributes(frc.G, "formanCurvature").values()), [1.5] * 3)

    # Diamond (two triangles sharing the edge (2,3)): all 4-cycles have a
    # diagonal inside and are discarded. Edge (2,3): two triangles -> 4/3;
    # the other edges: one triangle -> 5/6.
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)])
    frc = FormanRicci(G, method="balanced")
    frc.compute_ricci_curvature()
    npt.assert_array_almost_equal(
        list(nx.get_edge_attributes(frc.G, "formanCurvature").values()),
        [0.8333333333333334, 0.8333333333333334, 1.3333333333333333,
         0.8333333333333334, 0.8333333333333334])
    npt.assert_array_almost_equal(
        list(nx.get_node_attributes(frc.G, "formanCurvature").values()),
        [0.8333333333333334, 1.0, 1.0, 0.8333333333333334])

    # Two squares sharing the edge (2,4), plus leaves 3 and 5 to avoid other
    # 4-cycles: edge (2,4) has a genuine diagonal-free 4-cycle based at it
    # -> Ric = 2/2 + 2/2 - 2 + (1+1)/(1*2) = 1; edges (0,2), (1,4): one
    # 4-cycle each with d=(3,2) -> 1/3; leaf edges (0,3), (1,5) -> 0.
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4)])
    frc = FormanRicci(G, method="balanced")
    frc.compute_ricci_curvature()
    npt.assert_array_almost_equal(
        list(nx.get_edge_attributes(frc.G, "formanCurvature").values()),
        [0.0, 0.3333333333333333, 0.0, 0.3333333333333333, 0.0, 1.0])
    npt.assert_array_almost_equal(
        list(nx.get_node_attributes(frc.G, "formanCurvature").values()),
        [0.1111111111111111, 0.1111111111111111, 0.6666666666666666, 0.0,
         0.6666666666666666, 0.0])

    # Star S3: every edge is incident to a leaf -> 0 by Definition 1.
    G = nx.star_graph(3)
    frc = FormanRicci(G, method="balanced")
    frc.compute_ricci_curvature()
    npt.assert_array_almost_equal(
        list(nx.get_edge_attributes(frc.G, "formanCurvature").values()), [0.0] * 3)
    npt.assert_array_almost_equal(
        list(nx.get_node_attributes(frc.G, "formanCurvature").values()), [0.0] * 4)

