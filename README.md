# GraphRicciCurvature
A Python library to compute Discrete Ricci curvature, Ricci flow, and Ricci community on NetworkX graph.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saibalmars/GraphRicciCurvature/blob/master/notebooks/tutorial.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/saibalmars/GraphRicciCurvature/master?filepath=notebooks%2Ftutorial.ipynb)
[![PyPI version](https://badge.fury.io/py/GraphRicciCurvature.svg)](https://badge.fury.io/py/GraphRicciCurvature)
[![Build Status](https://github.com/saibalmars/GraphRicciCurvature/actions/workflows/build.yml/badge.svg)](https://github.com/saibalmars/GraphRicciCurvature/actions)
[![Documentation Status](https://readthedocs.org/projects/graphriccicurvature/badge/?version=latest)](https://graphriccicurvature.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/graphriccicurvature)](https://pepy.tech/project/graphriccicurvature)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

-----
This work computes the **Ollivier-Ricci Curvature**[Ni], **Ollivier-Ricci Flow**[Ni2,Ni3], **Forman-Ricci Curvature**(or **Forman curvature**)[Sreejith, Samal], and **Ricci community**[Ni3] detected by Ollivier-Ricci flow metric.

<p align="center">
<img src="https://github.com/saibalmars/GraphRicciCurvature/raw/master/doc/_static/rf-manifold.png" title="Manifold Ricci flow" width="300" >
<img src="https://github.com/saibalmars/GraphRicciCurvature/raw/master/doc/_static/karate_demo.png" title="karate club demo" width="500" >
</p>

Curvature is a geometric property to describe the local shape of an object. If we draw two parallel paths on a surface with positive curvature like a sphere, these two paths move closer to each other while for a negatively curved surface like a saddle, these two paths tend to be apart. Currently there are multiple ways to discretize curvature on graph, in this library, we include two of the most frequently used discrete Ricci curvature: **Ollivier-Ricci curvature** which is based on optimal transportation theory and **Forman-Ricci curvature** which is base on CW complexes.  

In [Ni], edge Ricci curvature is observed to play an important role in the graph structure. An edge with positive curvature represents an edge within a cluster, while a negatively curved edge tent to be a bridge within clusters. Also, negatively curved edges are highly related to graph connectivity, with negatively curved edges removed from a connected graph, the graph soon become disconnected.

Ricci flow is a process to uniformized the edge Ricci curvature of the graph. For a given graph, the Ricci flow gives a "Ricci flow metric" on each edge as edge weights, such that under these edge weights, the Ricci curvature of the graph is mostly equal everywhere. In [Ni3], this "Ricci flow metric" is shown to be able to detect communities.

Both Ricci curvature and Ricci flow metric can act as a graph fingerprint for graph classification. The different graph gives different edge Ricci curvature distributions and different Ricci flow metric.

Video demonstration of Ricci flow for community detection:
<p align="center">
<a href="https://youtu.be/QlENb_XlJ_8">
<img src="https://github.com/saibalmars/GraphRicciCurvature/raw/master/doc/_static/ricci_community.png" title="Ricci Community" width="600" >
</a>
</p>

## Package Requirement

* [NetworkX](https://github.com/networkx/networkx) >= 2.0 (Based Graph library)
* [NetworKit](https://github.com/kit-parco/networkit) >= 6.1 (Shortest path algorithm)
* [NumPy](https://github.com/numpy/numpy) (POT support)
* [POT](https://github.com/rflamary/POT) (For optimal transportation distance)
* [python-louvain](https://github.com/taynaud/python-louvain) (For faster modularity computation)



## Installation

### Installing via pip

```bash
pip3 install [--user] GraphRicciCurvature
```

- From version 0.4.0, NetworKit is required to compute shortest path for density distribution. If the installation of NetworKit failed, please refer to [NetworKit' Installation instructions](https://github.com/networkit/networkit#installation-instructions).

### Upgrading via pip

To run with the latest code for the best performance, upgrade GraphRicciCurvature to the latest version with pip: 
```bash
pip3 install [--user] --upgrade GraphRicciCurvature
``` 

## Getting Started
- Check the jupyter notebook tutorial on [nbviewer](https://nbviewer.jupyter.org/github/saibalmars/GraphRicciCurvature/blob/master/notebooks/tutorial.ipynb) or [github](notebooks/tutorial.ipynb) for a walk through for the basic usage of Ricci curvature, Ricci flow, and Ricci flow for community detection.
- Or you can run it in directly on [binder](https://mybinder.org/v2/gh/saibalmars/GraphRicciCurvature/master?filepath=notebooks%2Ftutorial.ipynb) (no account required) or [Google colab](https://colab.research.google.com/github/saibalmars/GraphRicciCurvature/blob/master/notebooks/tutorial.ipynb) (Faster but Google account required).
- Check the [Documentations](https://graphriccicurvature.readthedocs.io/en/latest/).
- **Try out [sample graphs](https://github.com/saibalmars/RicciFlow-SampleGraphs) with precomputed Ricci curvature/flow.** 

## Simple Example

```python
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

print("\n- Import an example NetworkX karate club graph")
G = nx.karate_club_graph()

print("\n===== Compute the Ollivier-Ricci curvature of the given graph G =====")
# compute the Ollivier-Ricci curvature of the given graph G
orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
orc.compute_ricci_curvature()
print("Karate Club Graph: The Ollivier-Ricci curvature of edge (0,1) is %f" % orc.G[0][1]["ricciCurvature"])

print("\n===== Compute the Forman-Ricci curvature of the given graph G =====")
frc = FormanRicci(G)
frc.compute_ricci_curvature()
print("Karate Club Graph: The Forman-Ricci curvature of edge (0,1) is %f" % frc.G[0][1]["formanCurvature"])

# -----------------------------------
print("\n=====  Compute Ricci flow metric - Optimal Transportation Distance =====")
G = nx.karate_club_graph()
orc_OTD = OllivierRicci(G, alpha=0.5, method="OTD", verbose="INFO")
orc_OTD.compute_ricci_flow(iterations=10)
print("\n=====  Compute Ricci community - by Ricci flow =====")
clustering = orc_OTD.ricci_community()

```

More example in [example.py](example.py).


----
## Related Works

- Curvature Graph Network (ICLR2020)  [openreview](https://openreview.net/forum?id=BylEqnVFDB), [code](https://github.com/yeze16159/CurvGN)


## Reference

[Ni]: Ni, C.-C., Lin, Y.-Y., Gao, J., Gu, X., and Saucan, E. "Ricci curvature of the Internet topology" (Vol. 26, pp. 2758–2766). Presented at the 2015 IEEE Conference on Computer Communications (INFOCOM), IEEE. [arXiv](https://arxiv.org/abs/1501.04138)

[Ni2]: Ni, C.-C., Lin, Y.-Y., Gao, J., and Gu, X. "Network Alignment by Discrete Ollivier-Ricci Flow", Graph Drawing 2018, [arXiv](https://arxiv.org/abs/1809.00320)

[Ni3]: Ni, C.-C., Lin, Y.-Y., Luo, F. and Gao, J. "Community Detection on Networks with Ricci Flow", Scientific Reports 9, 9984 (2019), [arXiv](https://arxiv.org/abs/1907.03993)

[Sreejith]: Sreejith, R. P., Karthikeyan Mohanraj, Jürgen Jost, Emil Saucan, and Areejit Samal. "Forman Curvature for Complex Networks." Journal of Statistical Mechanics: Theory and Experiment 2016 (6). IOP Publishing: 063206. [arxiv](https://arxiv.org/abs/1603.00386)

[Samal]: Samal, A., Sreejith, R.P., Gu, J. et al. "Comparative analysis of two discretizations of Ricci curvature for complex networks." Scientific Report 8, 8650 (2018). [arXiv](https://arxiv.org/abs/1712.07600)

## Contact

Please contact [Chien-Chun Ni](http://www3.cs.stonybrook.edu/~chni/).


## Cite

If you use this code in your research, please considering cite our paper:

```
@article{ni2019community,
  title={Community detection on networks with ricci flow},
  author={Ni, Chien-Chun and Lin, Yu-Yao and Luo, Feng and Gao, Jie},
  journal={Scientific reports},
  volume={9},
  number={1},
  pages={1--12},
  year={2019},
  publisher={Nature Publishing Group}
}
```