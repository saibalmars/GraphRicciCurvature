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
