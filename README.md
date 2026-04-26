#HACH‑MVC
Hierarchical Anchor‑Guided Contrastive Learning with Heterogeneous GNN for Multi‑View Clustering

This repository contains the official implementation of the method presented in the above paper.
It is released under the MIT license and is fully reproducible – the same results reported in the paper can be obtained by following the instructions below.

Overview
Modern datasets are rarely monolithic – images, text, graphs, and other modalities coexist and complement each other.
The task of Multi‑View Clustering (MVC) is to discover the underlying groups by jointly leveraging all views.
Existing MVC methods either:

------------------------------------------------------------------------------------------------------------------
|                Issue               |            Prior Approaches           |            	HACH‑MVC             |
------------------------------------------------------------------------------------------------------------------
|  Static, noisy graph construction  |  Heuristic k‑NN	Iterative refinement | view‑specific graph auto‑encoders |
------------------------------------------------------------------------------------------------------------------
|   Poor handling of heterogeneity	 |  Simple concatenation or weighted sum |         Heterogeneous GNN         |
------------------------------------------------------------------------------------------------------------------
| Sample imbalance & false negatives |       Adversarial augmentations       |    Hierarchical anchor‑guided     |
|     in contrastive learning        |                                       |       contrastive learning        |
------------------------------------------------------------------------------------------------------------------
|   Rigid K‑means initialization	 |         2‑stage pipelines	         |    Adaptive bi‑stage clustering   |
------------------------------------------------------------------------------------------------------------------


https://doi.org/10.1016/j.knosys.2026.116033
 
