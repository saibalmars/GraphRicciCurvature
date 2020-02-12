import networkx as nx

from GraphRicciCurvature.FormanRicci import FormanRicci


def test_compute_ricci_curvature():
    Gd = nx.DiGraph()
    Gd.add_edges_from([(1, 2), (2, 3), (3, 4), (2, 4), (4, 2)])
    orc_directed = FormanRicci(Gd)
    orc_directed.compute_ricci_curvature()

    frc = nx.get_edge_attributes(orc_directed.G, "formanCurvature")

    assert frc == {(1, 2): -2, (2, 3): 0, (2, 4): 0, (3, 4): 1, (4, 2): 0}
