# HACH‑MVC
Hierarchical Anchor‑Guided Contrastive Learning with Heterogeneous GNN for Multi‑View Clustering
https://doi.org/10.1016/j.knosys.2026.116033

This repository contains the official implementation of the method presented in the above paper.


# Highlights
•
A novel multi-view clustering framework, HACH-MVC, is proposed.

•
Introduces a novel hierarchical anchor-guided contrastive learning mechanism.

•
Adaptive receptive field balances noise and structure in anchor detection.

•
Proposes an adaptive bi-stage clustering module to avoid local optima.
•
Achieves high resilience to feature and structure noise.

# Overview

HACH-MVC is an end-to-end framework that constructs view-specific graphs, iteratively refines them through a heterogeneous GNN, and employs a learnable dual-fusion mechanism to derive a consensus graph and feature representation followed by a novel contrastive learning and bi-stage clustering methods. Its core innovation, an influence-aware, anchor-based hierarchical contrastive mechanism, uses anchor nodes identified via a novel adaptive receptive field for nested contrastive learning, preserving micro-group cohesion, enforcing macro-cluster separation. Finally, hybrid adaptive clustering transitions, overcome local optima. This strategic integration of the components enables comprehensive model learning and adaptation to the underlying data structures and the clustering task itself.

_________________________________________________________________________________________________________________
| ----------------Issue--------------- | -----------Prior Approaches------------ | --------------HACH‑MVC--------------- |
_________________________________________________________________________________________________________________
| -Static, noisy graph construction- | --Heuristic k‑NN	Iterative refinement-- | --view‑specific graph auto‑encoders-- |
_________________________________________________________________________________________________________________
| -Poor handling of heterogeneity- | Simple concatenation or weighted sum | ---------Heterogeneous GNN---------- |
_________________________________________________________________________________________________________________
| Sample imbalance, false negatives | -------Adversarial augmentations------ | ------ Hierarchical anchor‑guided----- | 
| -------in contrastive learning------ | -------------------------------------------| -----------contrastive learning--------- |
_________________________________________________________________________________________________________________
| ----Rigid K‑means initialization---- | ------------2‑stage pipelines----------- | ------Adaptive bi‑stage clustering----- |
_________________________________________________________________________________________________________________


## License

This project is licensed under a custom Non‑Commercial + Attribution license by Ali Takrar.
Any use, modification, or redistribution must credit the author and cite the
associated research paper:

HACH-MVC: Hierarchical Anchor-Guided Contrastive Learning with Heterogeneous GNN for Multi-View Clustering
https://doi.org/10.1016/j.knosys.2026.116033

Commercial use of this software—modified or unmodified—requires prior written
permission. Contact: takrar.co@gmail.com

 
