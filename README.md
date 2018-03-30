# GraphRicciCurvature
Compute Ricci curvature on NetworkX graph


-----
This work is to compute the **Ollivier-Ricci Curvature** as shown in the paper *Ricci Curvature of the Internet Topology*[Ni]. 
Ollivier-Ricci curvature of graph is an intrinsic geometric metric. An edge with positive curvature represents an edge within a cluster, while an negatively curved edge tents to be a bridge within clusters. 
<p align="center"> 
<img src="https://www3.cs.stonybrook.edu/~chni/img/3967-graph-gray-small.png">
</p>

## Package Requirement

* [networkx](https://github.com/networkx/networkx)
* [numpy](https://github.com/numpy/numpy)
* [cvxpy](https://github.com/cvxgrp/cvxpy)

## Example

```python
import networkx as nx
from GraphRicciCurvature.OllivierRicci import ricciCurvature

# import an example NetworkX karate club graph
G=nx.karate_club_graph()

# compute the Ricci curvature of the given graph G
G=ricciCurvature(G)
print("Karate Club Graph: The Ollivier-Ricci curvature of edge (0,1) is %f" % G[0][1]["ricciCurvature"])

# Construct a directed graph example
Gd = nx.DiGraph()
Gd.add_edges_from([(1, 2), (2, 3), (3, 4), (2, 4), (4, 2)])

# compute the Ricci curvature of the given directed graph Gd
Gd = ricciCurvature(Gd)
for n1, n2 in Gd.edges():
    print("Directed Graph: The Ollivier-Ricci curvature of edge(%d,%d) id %f" % (n1, n2, Gd[n1][n2]["ricciCurvature"]))

```

-----

- [Ni]: Ni, C.-C., Lin, Y.-Y., Gao, J., Gu, X., & Saucan, E. (2015). Ricci curvature of the Internet topology (Vol. 26, pp. 2758–2766). Presented at the 2015 IEEE Conference on Computer Communications (INFOCOM), IEEE. [arXiv](https://arxiv.org/abs/1501.04138)
- [Lin]: Lin, Y., Lu, L., & Yau, S.-T. (2011). Ricci curvature of graphs. Tohoku Mathematical Journal, 63(4), 605–627.
- [Ollivier]: Ollivier, Y. (2009). Ricci curvature of Markov chains on metric spaces. Journal of Functional Analysis, 256(3), 810–864.
