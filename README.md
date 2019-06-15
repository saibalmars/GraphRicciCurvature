# GraphRicciCurvature
Compute Discrete Ricci curvature and Ricci flow on NetworkX graph.


-----
This work computes the **Ollivier-Ricci Curvature**[Ni], **Ollivier-Ricci Flow**[Ni2] and **Forman-Ricci Curvature**(or **Forman curvature**)[Sreejith].

Curvature is a geometric property to describe the local shape of an object. 
If we draw two parallel paths on a surface with positive curvature like a sphere, these two paths move closer to each other while for a negative curved surface like saddle, these two paths tend to be apart.

To apply the Ricci curvature to every node and edge in graph, as in [Ni], we observe that the edge Ricci curvature play an important role in graph structure. An edge with positive curvature represents an edge within a cluster, while an negatively curved edge tents to be a bridge within clusters. Also, negatively curved edges are highly related to graph connectivity, with negatively curved edges removed from a connected graph, the graph soon become disconnected.

Ricci flow is a process to uniformized the edge Ricci curvature of the graph. For a given graph, the Ricci flow gives a "Ricci flow metric" on each edge as edge weights, such that under these edge weights, the Ricci curvature of the graph is mostly equal everywhere. 

Both Ricci curvature and Ricci flow metric can be act as a graph fingerprint. Different graph gives different edge Ricci curvature distributions and different Ricci flow metric. 


<p align="center"> 
<img src="https://www3.cs.stonybrook.edu/~chni/img/3967-graph-gray-small.png">
</p>

## Package Requirement

* [NetworkX](https://github.com/networkx/networkx) (Based Graph library)
* [CVXPY](https://github.com/cvxgrp/cvxpy) (Optimal transportation solver)
* [NumPy](https://github.com/numpy/numpy) (CVXPY support)
* [NetworKit](https://github.com/kit-parco/networkit) (*Optional: for faster parallel shortest path computation*)

## Example

```python
import networkx as nx
from GraphRicciCurvature.OllivierRicci import ricciCurvature
from GraphRicciCurvature.FormanRicci import formanCurvature
from GraphRicciCurvature.RicciFlow import compute_ricciFlow

# import an example NetworkX karate club graph
G = nx.karate_club_graph()

# compute the Ollivier-Ricci curvature of the given graph G
G = ricciCurvature(G, alpha=0.5, weight=None, verbose=False)
print("Karate Club Graph: The Ollivier-Ricci curvature of edge (0,1) is %f" % G[0][1]["ricciCurvature"])

# compute the Forman-Ricci curvature of the given graph G
G = formanCurvature(G, verbose=False)
print("Karate Club Graph: The Forman-Ricci curvature of edge (0,1) is %f" % G[0][1]["formanCurvature"])

#-----------------------------------
# Construct a directed graph example
Gd = nx.DiGraph()
Gd.add_edges_from([(1, 2), (2, 3), (3, 4), (2, 4), (4, 2)])

# compute the Ollivier-Ricci curvature of the given directed graph Gd
Gd = ricciCurvature(Gd)
for n1, n2 in Gd.edges():
    print("Directed Graph: The Ollivier-Ricci curvature of edge(%d,%d) id %f" % (n1, n2, Gd[n1][n2]["ricciCurvature"]))

# compute the Forman-Ricci curvature of the given directed graph Gd
Gd = formanCurvature(Gd)
for n1, n2 in Gd.edges():
    print("Directed Graph: The Forman-Ricci curvature of edge(%d,%d) id %f" % (n1, n2, Gd[n1][n2]["formanCurvature"]))

#-----------------------------------
# Multiprocessing computation is also supported
G=nx.random_regular_graph(8,1000)
ricciCurvature(G,proc=4)

# -----------------------------------
# Compute Ricci flow metric - Optimal Transportation Distance
G = nx.karate_club_graph()
G = compute_ricciFlow(G, iterations=10, method="OTD")

# Compute Ricci flow metric - Average Transportation Distance (Faster)
G = nx.karate_club_graph()
G = compute_ricciFlow(G, iterations=10, method="ATD")
```


-----
## Reference

[Ni]: Ni, C.-C., Lin, Y.-Y., Gao, J., Gu, X., and Saucan, E. (2015). "Ricci curvature of the Internet topology" (Vol. 26, pp. 2758–2766). Presented at the 2015 IEEE Conference on Computer Communications (INFOCOM), IEEE. [arXiv](https://arxiv.org/abs/1501.04138)

[Ni2]: Ni, C.-C., Lin, Y.-Y., Gao, J., and Gu, X. . "Network Alignment by Discrete Ollivier-Ricci Flow", Graph Drawing 2018, [arXiv](https://arxiv.org/abs/1809.00320)

[Ollivier]: Ollivier, Y. (2009). "Ricci curvature of Markov chains on metric spaces". Journal of Functional Analysis, 256(3), 810–864.

[Forman]: Forman. 2003. "Bochner’s Method for Cell Complexes and Combinatorial Ricci Curvature." Discrete & Computational Geometry 29 (3). Springer-Verlag: 323–74.

[Sreejith]: Sreejith, R. P., Karthikeyan Mohanraj, Jürgen Jost, Emil Saucan, and Areejit Samal. 2016. “Forman Curvature for Complex Networks.” Journal of Statistical Mechanics: Theory and Experiment 2016 (6). IOP Publishing: 063206. [arxiv](https://arxiv.org/abs/1603.00386)


-----
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