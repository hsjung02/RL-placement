# RL-placement
RL-placement is an automated framework for RL-based chip placement.

## File tree

```
📦main
 ┣ 📂netlist
 ┃ ┣ 📜adjacency_matrix
 ┃ ┣ 📜cells
 ┃ ┣ 📜HGraphFile.hgr.part.100
 ┃ ┣ 📜HGraphFile.hgr.part.4
 ┃ ┣ 📜HGraphFile.txt
 ┃ ┣ 📜ispd18_test3.def
 ┃ ┣ 📜ispd18_test3.lef
 ┃ ┣ 📜macro_indices
 ┃ ┗ 📜std_indices
 ┣ 📜parsing.ipynb
 ┣ 📜placement_ispd18test3.ipynb
 ┣ 📜placement_ispd18test3.pt
```

## Dependencies

| Module | Version |
| --- | --- |
| python | 3.9.12 |
| torch | 1.12.1 |
| numpy | 1.23.4 |
| matplotlib | 3.6.1 |

## Run

1. Run parsing.ipynb
2. Run placement_ispd18test3.ipynb

## References

1. [Mirhoseini, A., et al. (2020). Chip Placement with Deep Reinforcement Learning, arXiv:2004.10746(doi: https://doi.org/10.48550/arXiv.2004.10746)](https://arxiv.org/abs/2004.10746)
2. [Yue, S., et al. (2022). Scalability and Generalization of Circuit Training for Chip Floorplanning. Proceedings of the 2022 International Symposium on Physical Design. Virtual Event, Canada, Association for Computing Machinery: 65–70.](https://dl.acm.org/doi/abs/10.1145/3505170.3511478)
3. https://github.com/google-research/circuit_training.git repository
