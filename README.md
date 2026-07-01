# GCN & GAT from Scratch

Pure-Python (no `torch`, no `torch_geometric`) implementations of **Graph Convolutional Networks (GCN)** and **Graph Attention Networks (GAT)**, built from first principles to understand what's happening under the hood before using library implementations.

This notebook walks through the math and code for both layer types, then runs a toy 3-node graph through each model to sanity-check the forward pass.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shris2810/link_prediction/blob/main/GCN_and_GAT.ipynb)

## Contents

- **GAT (Graph Attention Network)**
  - Single attention head implementation (linear transform → attention scores → softmax → weighted aggregation)
  - Multi-head attention (concatenation across heads)
  - 2-layer `GATModel`
- **GCN (Graph Convolutional Network)**
  - Degree-normalized layer (`NormalizedGCNLayer`) following the standard symmetric normalization: Â = D̂⁻¹ᐟ² (A + I) D̂⁻¹ᐟ²
  - Simple mean-aggregation layer (`GCNLayer`) for comparison
  - 2-layer models for both variants: `TwoHopNormalizedGCN`, `TwoHopGCN`
- **Toy example**: A 3-node fully-connected graph with 2D node features, used to verify each model produces sane output embeddings

## Key Concepts Covered

| Concept | GCN | GAT |
|---|---|---|
| Neighbor weighting | Fixed (degree-based normalization or simple mean) | Learned (attention scores per edge) |
| Self-loops | Added explicitly (`A = A + I`) | Included as another neighbor in attention |
| Aggregation | `H = ÂXW` | `H_i = Σ α_ij · (Wh_j)` |
| Multi-head support | No | Yes (concat on hidden layers, average on output layer) |

## Structure

```
GCN_and_GAT.ipynb
├── GAT layer                  # theory notes + GATHead, MultiHeadGAT, GATModel
├── GCN layer                  # theory notes + GCNLayer, NormalizedGCNLayer, Two-hop models
└── Running Models             # forward pass on toy graph, printed embeddings
```

## Usage

Open the notebook in Colab (badge above) or locally with Jupyter, and run all cells top to bottom. Each model class exposes a `.forward(X, adj)` method:

```python
X = {0: [1.0, 0.0], 1: [0.0, 1.0], 2: [1.0, 1.0]}   # node features
adj = {0: [1, 2], 1: [0, 2], 2: [0, 1]}              # adjacency list

model = TwoHopNormalizedGCN()
embeddings = model.forward(X, adj)
```

## Notes

- Weights are randomly initialized (`random.uniform(-0.1, 0.1)`) and untrained — this notebook demonstrates the **forward pass mechanics only**, not learned representations.
- Built as a learning exercise before moving to `torch_geometric` for production-scale link prediction (see related work: GCN/GAT-based link prediction with ComplEx embeddings, evaluated via MRR / Hits@K / ROC-AUC).

## Requirements

- Python 3 standard library only (`math`, `random`) — no external dependencies needed to run this notebook.
