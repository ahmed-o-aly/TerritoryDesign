# 1. TerritoryDesign
This directory contains materials related to a paper on Delivery Territory Design Problem (DTDP) using a stochastic Variable Neighborhood Search (SVNS). The directory contains benchmark instances of two types along with a notebook to guide the user on how to generate their own graphs.

---
## 2. Types of graphs
### 2.1. T Graph Instances
These graph types are plannar graphs that were generated using Delaunay triangulation. 

Each node contains 3 node attributes:
1. n_customers: Number of customers in a given node where $n_{customers}=[4,20]$
2. demand: Demand in a given node where $demand=[15,400]$
3. workload: Workload in a given node where $workload=[15,100]$

### 2.2. G Graph Instances
These graph types are grid-like graphs generated using an $N \times N$ grid and had nodes systematically removed while maintaining connectivity. 

Each node contains 3 node attributes:
1. n_customers: Number of customers in a given node where $n_{customers}=[4,20]$
2. demand: Demand in a given node where $demand=[15,400]$
3. workload: Workload in a given node where $workload=[15,100]$

Each edge contains 1 distance attribute $distance = [5,12]$

---
## 3. File Directory
```bash
TerritoryDesign
├───GGraphInstances
│   └───newGeneratedInstances
│       ├───27x27Graphs
│       ├───30x30Graphs
│       └───33x33Graphs
├───TGraphInstances
├───CITATION.cff
├───generateGraphs.ipynb
└───README.md
```
