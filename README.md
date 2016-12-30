# GraphRicciCurvature
Compute Ricci curvature on NetworkX graph


-----
This work is to compute the **Ollivier-Ricci Curvature** as shown in the paper *Ricci Curvature of the Internet Topology*[Ni]. 



## Package Requirement

* networkx
* numpy
* cvxpy

## Example

```python
import networkx as nx
from GraphRicciCurvature.OllivierRicci import ricciCurvature

# import an example NetworkX karate club graph
G=nx.karate_club_graph()

# compute the ricci curvature of the given graph G
G=ricciCurvature(G)
print G[0][1]["ricciCurvature"]

```

-----

- [Ni]: Ni, C.-C., Lin, Y.-Y., Gao, J., David Gu, X., & Saucan, E. (n.d.). Ricci curvature of the Internet topology (pp. 2758–2766). Presented at the IEEE INFOCOM 2015 - IEEE Conference on Computer Communications, IEEE. http://doi.org/10.1109/INFOCOM.2015.7218668
