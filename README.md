# Delivery Territory Design Problem (DTDP)

Materials supporting the open-access paper on stochastic Variable Neighborhood Search for the DTDP (https://doi.org/10.1016/j.cor.2024.106756). The repository provides benchmark instances, algorithm implementations, and experiment outputs so you can reuse or extend the study.

## Contents
- `DTDPAlgorithms.py` — Python implementation of the core heuristics (construction, local search, BVNS, path relinking).
- `TGraphInstances/` — 30 planar benchmark graphs generated via Delaunay triangulation (10 each with 500, 600, 700 nodes).
- `GGraphInstances/newGeneratedInstances/` — 90 grid-derived graphs (30 each in `27x27Graphs`, `30x30Graphs`, `33x33Graphs`).
- `generateGraphs.ipynb` — notebook used to generate new T and G instances.
- `Results/` — JSON logs from the paper’s experiments (MIP, VNS, PR variants).
- `CITATION.cff` — citation metadata for this dataset and code.
- License: Creative Commons CC-BY-NC. Cite with the DOI above or `CITATION.cff`.

## Environment and dependencies
- Tested on Python 3.9. 
- Minimal packages: `networkx`, `numpy`, `scipy`.
- Useful extras: `pandas`, `matplotlib`, `jupyter`.
```bash
pip install networkx numpy scipy pandas matplotlib jupyter
```

## Instance format (GraphML)
- Nodes (all): `n_customers` ∈ [4,20], `demand` ∈ [15,400], `workload` ∈ [15,100], coordinates `x`, `y`.
- T graphs: Planar, unweighted edges.
- G graphs: Grid-derived, edges carry `distance` ∈ [5,12].
- Summary by folder:

| Folder | Graphs | Nodes | Edge count range |
| --- | --- | --- | --- |
| `TGraphInstances`| 30 | 500-700 | 1465–2071 |
| `GGraphInstances/newGeneratedInstances/27x27Graphs` | 30 | 486 | 609–648 |
| `GGraphInstances/newGeneratedInstances/30x30Graphs` | 30 | 600 | 758–803 |
| `GGraphInstances/newGeneratedInstances/33x33Graphs` | 30 | 726 | 922–969 |

## Quick start
1) Load an instance:
```python
import networkx as nx
G = nx.read_graphml("TGraphInstances/planar500_G0.graphml")
print(len(G), "nodes", len(G.edges), "edges")
```

2) Run the BVNS heuristic:
```python
from DTDPAlgorithms import TerritoryDesignProblem, BVNS

tdp = TerritoryDesignProblem(
    graph_input=G,
    delta=0.05,          # balance tolerance between districts
    llambda=0.4,         # weight between dispersion vs. balance
    rcl_parameter=0.2,   # restricted candidate list threshold
    nr_districts=10
)

bvns = BVNS(tdp_instance=tdp, shaking_steps=25, fail_max=50, nrInitSolutions=50)
obj_hist, inf_hist, best_solution, timeline = bvns.performBVNS()
print("Best objective:", obj_hist[-1], "Infeasibility:", inf_hist[-1])
```

3) Plot the solution by district:
```python
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_districts(G, districts, pos=None):
    if pos is None:
        pos = {n: (float(G.nodes[n]['x']), float(G.nodes[n]['y'])) for n in G.nodes}
    palette = cm.get_cmap("tab20")
    plt.figure(figsize=(8, 8))
    for k, nodes in districts.items():
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodes,
            node_color=[palette(k % 20)],
            node_size=12,
            label=f"District {k}"
        )
    nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.axis("off")
    plt.show()

districts = best_solution if isinstance(best_solution, dict) else best_solution["Districts"]
plot_districts(G, districts)
```

## Generate new graphs
Open `generateGraphs.ipynb` and run the notebook cells:
- T graphs: planar layouts built from Delaunay triangulation; adjust node counts and attribute ranges as needed.
- G graphs: start from an N×N grid, remove nodes while keeping connectivity, and assign the same attributes used in the paper.

## Results folder structure
- `VNSExperiments*`: BVNS timelines per instance. Keys: `objective`, `infeasibility`, `merit`, `lambda`, `time`.
- `PRExperiments*`: path relinking runs and best merit per instance.
- `MIPExperiments*`: mixed-integer programming baselines.

## Cite
If you use the instances or code, cite the paper above and `CITATION.cff`:
- Aly, A., Gabor, A. F., Sleptchenko, A. (2024). Delivery Territory Design Problem. Computers & Operations Research. https://doi.org/10.1016/j.cor.2024.106756
