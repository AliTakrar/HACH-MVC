# HACH‑MVC
Hierarchical Anchor‑Guided Contrastive Learning with Heterogeneous GNN for Multi‑View Clustering
https://doi.org/10.1016/j.knosys.2026.116033

This repository contains the official implementation of the method presented in the above paper.


Modern datasets are rarely monolithic – images, text, graphs, and other modalities coexist and complement each other.
The task of Multi‑View Clustering (MVC) is to discover the underlying groups by jointly leveraging all views.


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



 
