'''
@Project: HACH-MVC
@Author : Ali Takrar
@Email  : takrar.co@gmail.com
@File   : test_train
@Desc
    main function 
'''
import os
import nni
import time
import warnings
warnings.filterwarnings('ignore')
import argparse
import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop
from sklearn.cluster import KMeans
from utils.ops_al import target_distribution
from copy import deepcopy
from models.hach_mvc import HACHMVC
from utils.ops_pt import pretraining
from utils.ops_ev import get_evaluation_results
from utils.load_data_clusterft import load_data_clusterft

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster._supervised import check_clusterings
from scipy.spatial.distance import pdist, cdist


import torch
import torch.nn.functional as F
import math
import warnings

import numpy as np
import torch
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
import torch.nn.functional as F

def get_trend_slope(history, window=5):
    """
    Returns the slope of the last 'window' epochs.
    Negative slope = Improving (Metric is going down).
    Positive slope = Worsening (Metric is going up).
    """
    if len(history) < 2:
        return 0.0
    
    # Get the last N points
    y = np.array(history[-window:])
    x = np.arange(len(y))
    
    # Fit a line (y = mx + c), index [0] is the slope (m)
    slope, _ = np.polyfit(x, y, 1)
    
    # Normalize slope by the mean value to make it scale-invariant
    # (A drop of 0.1 is huge if value is 0.2, but tiny if value is 100)
    avg_val = np.mean(y) + 1e-6
    normalized_slope = slope / avg_val
    
    return normalized_slope

def get_volatility_penalty(history, window=5):
    """
    Calculates relative instability using Coefficient of Variation.
    Works for any scale of loss (0.001 or 1000).
    """
    if len(history) < window:
        return 0.0

    # Get recent window
    recent = np.array(history[-window:])
    
    # Calculate stats
    mu = np.mean(recent) + 1e-9 # Avoid division by zero
    sigma = np.std(recent)
    
    # Coefficient of Variation (CV)
    # This represents noise as a percentage of the signal
    cv = sigma / mu 
    
    # Logic:
    # If noise is < 10% (0.1), we consider it stable (penalty 0).
    # If noise is > 10%, we start penalizing.
    # We cap the penalty at 0.15 to avoid destroying confidence completely.
    
    if cv > 0.1: 
        # Linearly map CV to a penalty
        # Example: CV 0.2 (20% noise) -> Penalty 0.05
        penalty = (cv - 0.1) * 0.5 
        return min(0.15, penalty) # Cap max penalty
    
    return 0.0
    
def update_influential_nodes(adj_matrix, davies_bouldin_history ,loss_cli_history , epoch):
    num_nodes = adj_matrix.shape[0]
    min_influential_threshold = (np.log10(num_nodes) * np.log(num_nodes)) - np.log(num_nodes)
    if (epoch == 0): 
        print(f"Target Minimum Unique Influential Nodes: {min_influential_threshold:.4f}")
    device = adj_matrix.device
    #print('num_nodes: ' , num_nodes )
    
    # --- 0. Calculate Confidence (Quality + Gradient Trend) ---
    confidence = 0.5 # Neutral start

    if len(davies_bouldin_history) > 0 and len(loss_cli_history) > 0:
        
        # A. Static Score (How good are the absolute numbers?)
        curr_dbi = davies_bouldin_history[-1]
        curr_loss = loss_cli_history[-1]
        
        # Normalized to 0-1 (Assuming lower is better)
        score_dbi = 1.0 / (1.0 + curr_dbi)
        score_loss = 1.0 / (1.0 + curr_loss)
        static_score = (score_dbi + score_loss) / 2.0

        # B. Trend Score (Are we moving in the right direction?)
        # We calculate the slope of the last 5 epochs
        slope_dbi = get_trend_slope(davies_bouldin_history, window=5)
        slope_loss = get_trend_slope(loss_cli_history, window=5)
        
        # Average the slopes
        avg_slope = (slope_dbi + slope_loss) / 2.0
        
        # C. Interpret the Slope
        # If slope is negative (e.g. -0.1), we are improving -> Add to confidence
        # If slope is positive (e.g. +0.05), we are worsening -> Subtract
        # We invert the sign because negative slope is GOOD for Loss/DBI
        trend_bonus = -avg_slope * 2.0  # Multiply by factor to adjust sensitivity
        
        # Clamp trend bonus so it doesn't dominate too much (e.g., max +/- 0.2)
        trend_bonus = max(-0.2, min(0.2, trend_bonus))

        # D. Scale-Invariant Stability Check
        stability_penalty = 0.0
        if len(loss_cli_history) >= 5:
            stability_penalty = get_volatility_penalty(loss_cli_history, window=5)

        # E. Final Calculation
        confidence = static_score + trend_bonus - stability_penalty
        confidence = max(0.0, min(1.0, confidence))
    # --- 1. Adapt ft_k (Neighbors) and ft_h (Hops) ---
    
    # Base heuristic: log2(N)
    base_k = args.ft_k
    base_h = args.ft_h
    if(args.ft_dahk == 1 ):
        #print('confidence: ' , confidence)
        if epoch > 5 : 
            
            if confidence > 0.8:  # Clean Graph
                if(base_k == 7):
                    h = min(5, base_h + 1)
                    k = min(7, base_k - 1) 
                # Expand k: Trust more neighbors
                else:
                    k = min(7, base_k + 1) 
                
            elif confidence < 0.6: # Noisy Graph
                if(base_k == 1):
                    if(base_h != 1):
                        h = max(1, base_h - 1)
                        k = max(1, base_k + 1)
                    else: 
                        h = base_h
                        k = base_k
                # Limit k: Trust only the very strongest connection
                else:
                    k = max(1, base_k - 1)
            else:
                k = base_k
                    
        
            if confidence > 0.8 and confidence < 0.9: # Clean Graph
                if(base_h == 5):
                    k = min(7, base_k + 1)
                    h = min(5, base_h - 1)
                # Deep Field: Look 3-4 hops away to find global leaders
                else:
                    h = min(5, base_h + 1)
            elif confidence > 0.6 and confidence < 0.7: # Noisy Graph
                if(base_h == 1):
                    if(base_k != 1):
                        k = max(1, base_k - 1)
                        h = max(1, base_h + 1) 
                    else: 
                        h = base_h
                        k = base_k
                # Shallow Field: Only look at immediate neighbors to avoid noise accumulation
                else:
                    h = max(1, base_h - 1) 
            else:
                h = base_h # Default conservative depth
        else:
            k = args.ft_k
            h = args.ft_h
    else:    
        k = args.ft_k
        h = args.ft_h
    # Calculate normalized degree centrality
    #high_degree_norm = torch.sum(adj_matrix, dim=1) / len(adj_matrix.shape)
    high_degree_norm = torch.sum(adj_matrix, dim=1) / (num_nodes - 1 + 1e-9)  # normalize by possible neighbors
    '''
    adj_matrix1 = adj_matrix.clone()
    adj_matrix1.fill_diagonal_(-float('inf'))

    # Get the top-k highest affinities for each node
    topk_affinities, topk_indices = torch.topk(adj_matrix1, k, dim=1)
    '''
    mask = torch.eye(num_nodes, device=device).bool()
    adj_matrix1 = adj_matrix.clone()
    adj_matrix1[mask] = -1e12
    topk_affinities, topk_indices = torch.topk(adj_matrix1, k, dim=1)

    # Create a new adjacency matrix for top-k edges
    new_adj_matrix = torch.zeros_like(adj_matrix1)
    for i in range(num_nodes):
        for idx in topk_indices[i]:
            new_adj_matrix[i, idx] = 1
            new_adj_matrix[idx, i] = 1  # Ensure symmetry


    #high_degree_norm = torch.sum(new_adj_matrix, dim=1) / len(adj_matrix.shape)
    # Convert to numpy for connected components computation
    new_adj_matrix_np = new_adj_matrix.cpu().numpy()
    
    # Find nodes within 3 hops based on the adjacency matrix
    # A slightly more conventional and memory-efficient version
    csr_adj_matrix = csr_matrix(new_adj_matrix_np)


    while True:
        #print('----------------------- k : ' , k )
        args.ft_k = k
        #print('----------------------- h : ' , h )
        args.ft_h = h
        #reachable_hops = reachable_hops.astype(int)
        current_hop_matrix = csr_adj_matrix
        reachable_hops = (current_hop_matrix > 0).astype(int)
    
        # 2. Loop to generate subsequent hops (A^2, A^3 ... A^k)
        # We loop (args.ft_h - 1) times because we already have the 1st hop
        for _ in range(args.ft_h - 1):
            # Calculate next power: A^n = A^(n-1) @ A^1
            current_hop_matrix = current_hop_matrix @ csr_adj_matrix
            
            # Accumulate the reachability mask
            reachable_hops += (current_hop_matrix > 0).astype(int)
        
        # Identify the most influential node for each node within 3 hops
        influential_nodes = []
        most_influentials = []
        for node in range(num_nodes):
            within_h_hops = np.where(reachable_hops[node].toarray().flatten() > 0)[0]
            if len(within_h_hops) > 0:
                # Determine the most influential node
                node_centrality = {n: high_degree_norm[n].item() for n in within_h_hops}
                most_influential = max(node_centrality, key=node_centrality.get)

                if node != most_influential : 
                    influential_nodes.append((node, most_influential))
                    most_influentials.append(most_influential)
    
        
            
        unique_influentials = np.unique(most_influentials)
        count_unique = len(unique_influentials)
        # Check Threshold Condition
        if count_unique < min_influential_threshold:
            if h > 2 and k < 5: 
                    h -= 1
            elif k > 2 :
                    k -= 1
            else:
                    break
        else:
            # Threshold met
            break

    if epoch == 0:
        diameter = graph_diameter(new_adj_matrix_np)
        print('Graph Diameter:', diameter)
    if (base_k != args.ft_k or base_h != args.ft_h):
        args.ft_ch = 1
    return influential_nodes, unique_influentials

# Placeholder function for graph diameter (implement as needed)
def graph_diameter(adj_matrix_np):
    from scipy.sparse.csgraph import dijkstra
    dist_matrix = dijkstra(csgraph=adj_matrix_np, directed=False)
    finite_distances = dist_matrix[np.isfinite(dist_matrix)]
    return finite_distances.max()

# Internal constants for positive distance fallback
_STD_TOLERANCE_RATIO = 0.1
_FALLBACK_MIN_DIST_FRACTION = 0.05
_NUMERICAL_FLOOR = 1e-6
# Internal constant for exponential repulsion temperature stability
_MIN_TEMPERATURE = 0.01 # Minimum value for adaptive temperature tau


def contrastive_loss_influential(node_embeddings, influential_nodes, most_influentials, num_epochs, epoch):
    """
    Auto-adaptive contrastive loss using a unified conditional loss for positive pairs.
    This version applies attraction only when distance > delta_min.

    Args:
        node_embeddings (torch.Tensor): Node embeddings of shape (N, D).
        influential_nodes (list): List of tuples [(node, inf_node), ...].
        most_influentials (list or array-like): List or array of influential nodes (indices).
        num_epochs (int): Total number of training epochs.
        epoch (int): Current epoch number.

    Returns:
        torch.Tensor: Contrastive loss value.
    """
    # <<< CHANGE: Replaced separate positive losses with a single unified one >>>
    positive_unified_loss = torch.tensor(0.0, device=node_embeddings.device)
    min_pos_distance = torch.tensor(_NUMERICAL_FLOOR, device=node_embeddings.device)
    num_pos_pairs = 0

    # --- Positive Pair Processing (Unified Conditional Loss) ---
    if influential_nodes:
        influential_pairs = torch.tensor(influential_nodes, dtype=torch.long, device=node_embeddings.device)
        source_nodes = influential_pairs[:, 0]
        target_nodes = influential_pairs[:, 1]
        num_pos_pairs = len(influential_nodes)

        if num_pos_pairs > 0:
            pos_distances = F.pairwise_distance(node_embeddings[source_nodes], node_embeddings[target_nodes])
            
            # --- Adaptive Margin (delta_min) Calculation (Unchanged) ---
            with torch.no_grad():
                current_mean_pos_dist = pos_distances.mean()
                safe_mean_pos_dist = torch.max(current_mean_pos_dist, torch.tensor(_NUMERICAL_FLOOR, device=node_embeddings.device))
                current_std_pos_dist = pos_distances.std()
                if torch.isnan(current_std_pos_dist) or num_pos_pairs <= 1:
                    current_std_pos_dist = torch.tensor(0.0, device=node_embeddings.device)

                is_std_too_small = current_std_pos_dist < (_STD_TOLERANCE_RATIO * safe_mean_pos_dist)

                if is_std_too_small:
                    calculated_min_dist = _FALLBACK_MIN_DIST_FRACTION * safe_mean_pos_dist
                else:
                    calculated_min_dist = safe_mean_pos_dist - current_std_pos_dist

                min_pos_distance = torch.max(
                    torch.relu(calculated_min_dist),
                    torch.tensor(_NUMERICAL_FLOOR, device=node_embeddings.device)
                )

            # <<< CHANGE: Implemented the new unified conditional loss formula >>>
            # This single line replaces both attraction and repulsion for positive pairs.
            # It penalizes distances only when they are greater than the minimum margin.
            positive_unified_loss = F.relu(pos_distances - min_pos_distance.detach()).mean()

    # --- Negative Pair Processing (Enhanced Push) ---
    # This entire section remains unchanged as the logic for negative pairs is separate.
    neg_loss_margin = torch.tensor(0.0, device=node_embeddings.device)
    neg_loss_push = torch.tensor(0.0, device=node_embeddings.device)
    neg_weight_ = torch.tensor(1.0, device=node_embeddings.device)
    num_neg_pairs_used = 0

    is_most_influentials_valid = False
    if most_influentials is not None:
        if isinstance(most_influentials, torch.Tensor):
             if most_influentials.numel() > 0: is_most_influentials_valid = True
        elif hasattr(most_influentials, '__len__') and len(most_influentials) > 0:
             is_most_influentials_valid = True

    if is_most_influentials_valid:
        if not isinstance(most_influentials, torch.Tensor):
            most_influentials_tensor = torch.tensor(most_influentials, dtype=torch.long, device=node_embeddings.device)
        else:
            most_influentials_tensor = most_influentials.to(device=node_embeddings.device, dtype=torch.long)

        if most_influentials_tensor.numel() > 0:
            unique_influential_indices = torch.unique(most_influentials_tensor)

            if len(unique_influential_indices) > 1:
                influential_embeds = node_embeddings[unique_influential_indices]
                neg_distances_matrix = torch.cdist(influential_embeds, influential_embeds, p=2)
                mask = torch.triu(torch.ones_like(neg_distances_matrix), diagonal=1).bool()
                neg_distances = neg_distances_matrix[mask]

                if neg_distances.numel() > 0:
                    num_neg_pairs_used = neg_distances.numel()
                    safe_neg_distances = torch.clamp(neg_distances, min=_NUMERICAL_FLOOR)

                    with torch.no_grad():
                        margin_mean = safe_neg_distances.mean()
                        margin_std = safe_neg_distances.std()
                        if torch.isnan(margin_std) or safe_neg_distances.numel() <= 1: margin_std = torch.tensor(0.0)
                        margin = margin_mean + margin_std
                        if torch.isnan(margin): margin = torch.tensor(1.0, device=node_embeddings.device)
                        temperature = torch.max(margin_mean, torch.tensor(_MIN_TEMPERATURE, device=node_embeddings.device))
                        
                    neg_loss_margin = F.relu(margin.detach() - safe_neg_distances).mean()
                    neg_loss_push = torch.exp(-safe_neg_distances / temperature.detach()).mean()

                    with torch.no_grad():
                        max_neg_dist = safe_neg_distances.max()
                        neg_weight_floor = 1.5
                        current_neg_weight = neg_weight_floor

                        if max_neg_dist > 1e-6 :
                            decay_target_epoch = max(1.0, num_epochs / 10.0)
                            log_arg = max(max_neg_dist.item(), 1.0)
                            if decay_target_epoch > 0:
                                decay_rate = math.log(log_arg) / decay_target_epoch
                                current_neg_weight = (max(0.0, max_neg_dist.item() - neg_weight_floor)) * math.exp(-decay_rate * epoch) + neg_weight_floor
                            else:
                                warnings.warn("decay_target_epoch is zero, using default neg_weight_floor.")
                        neg_weight_ = torch.tensor(current_neg_weight, device=node_embeddings.device)

    # <<< CHANGE: Simplified the total loss calculation >>>
    # Removed pos_attraction_loss and pos_repulsion_loss.
    total_loss = positive_unified_loss + \
                 neg_weight_ * (neg_loss_margin + neg_loss_push)


    # --- Normalization ---
    total_pairs = num_pos_pairs + num_neg_pairs_used
    if total_pairs > 0:
         return torch.clamp(total_loss / total_pairs, min=0.0)
    else:
        return torch.tensor(0.0, device=node_embeddings.device, requires_grad=node_embeddings.requires_grad)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)



params = nni.get_next_parameter()

params = {'pt_pm_lr': 0.0001  ,'pm_knns': 25, 'seed': 42, 'ft_lr': 1e-5, 'final_lr' : 1e-6 , 'ft_sp_weight': 0.5,
          'pt_pm_sp_weight': 0.1, 'ft_num_epochs': 401 , 'pt_pm_num_epochs': 800, 'n_repeated': 10,
          'ft_update_interval': 1 , 'ft_cl_weight': 1.0, 'ft_pl_weight': 1.0, 'ft_cli_weight': 1.0,
          'pm_struct_noise_level': 0.0 ,'pm_feature_noise_level':0.0 , 'ft_dahk' : 1 ,'ft_h': 3 ,'ft_k': 3 , 'ft_ch' : 0 }

print('params',params)


def del_files(path_file):
    # Check if the directory exists
    if os.path.exists(path_file) and os.path.isdir(path_file):
        try:
            # List all items in the directory
            ls = os.listdir(path_file)
            for item in ls:
                f_path = os.path.join(path_file, item)
                # Check if it is a file and delete it
                if os.path.isfile(f_path):
                    os.remove(f_path)
                    print(f"Deleted file: {f_path}")
                # Optionally check if it's a directory and delete it
                elif os.path.isdir(f_path):
                    os.rmdir(f_path)  # Use shutil.rmtree() for non-empty directories
                    print(f"Deleted directory: {f_path}")
        except Exception as e:
            print(f"An error occurred while trying to delete files: {e}")
    else:
        print(f"Directory does not exist: {path_file}")
    

if __name__ == '__main__':
    # Configuration settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=str, default=params['seed'], help='The number of cuda device.')
    parser.add_argument('--n_repeated', type=int, default=params['n_repeated'],help='Number of repeated experiments')  # 50
    parser.add_argument('--pm_struct_noise_level', type=float, default=params['pm_struct_noise_level'],help='Percentage of structrual noise')  # 0.0
    parser.add_argument('--pm_feature_noise_level', type=float, default=params['pm_feature_noise_level'],help='Percentage of feature noise')  # 0.0
    parser.add_argument('--ft_dahk', type=float, default=params['ft_dahk'],help='dynamicaly adjust h & k')  # 1
    parser.add_argument('--ft_h', type=float, default=params['ft_h'],help='h hop for detect LIA')  # 3
    parser.add_argument('--ft_k', type=float, default=params['ft_k'],help='TopK for detect LIA')  # 3
    parser.add_argument('--ft_ch', type=float, default=params['ft_ch'],help='alow to change cli_w pl_w sp_w')  # 3
    parser.add_argument('--ft_sp_weight', type=float, default=params['ft_sp_weight'], help='weight of structure preservation loss')  # 0.001
    parser.add_argument('--ft_cl_weight', type=float, default=params['ft_cl_weight'], help='weight of clustering loss')
    parser.add_argument('--ft_pl_weight', type=float, default=params['ft_pl_weight'], help='weight of clustering loss')
    parser.add_argument('--ft_cli_weight', type=float, default=params['ft_cli_weight'], help='weight of clustering loss')
    parser.add_argument('--pm_knns', type=int, default=params['pm_knns'],help='the number of nearest neighbors in PM')  # 50 params['pm_knns']
    parser.add_argument('--ft_update_interval', type=int, default=params['ft_update_interval'],help='weight of structure preservation loss')  # 50
    parser.add_argument('--pt_pm_lr', type=float, default=params['pt_pm_lr'],help='learning rate in pretraining stage.')  # 0.00001 params['pt_pm_lr']
    parser.add_argument('--pt_pm_num_epochs', type=int, default=params['pt_pm_num_epochs'],help='number of layer-wise training epochs.')  # 500
    parser.add_argument('--ft_lr', type=float, default=params['ft_lr'],help='learning rate in pretraining stage.')  # params['pt_pm_lr']0.00001
    parser.add_argument('--final_lr', type=float, default=params['final_lr'],help='final learning rate .')  # 0.000001 params['pt_pm_lr']
    parser.add_argument('--ft_num_epochs', type=int, default=params['ft_num_epochs'],help='number of layer-wise training epochs.')  # 1000


    parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--load_saved', action='store_true', default=False, help='load saved data.')
    parser.add_argument('--cuda_device', type=str, default='0', help='The number of cuda device.')
    parser.add_argument('--direction', type=str, default='./data/datasets/', help='direction of datasets')
    parser.add_argument('--dataset_name', type=str, default='MSRC_v1', help='The dataset used for training/testing')
    parser.add_argument('--normalization', type=str, default='normalize', help='default normalize')
    parser.add_argument('--am_first_dim', type=int, default=256, help='the dim of the first layer in PM')
    parser.add_argument('--am_second_dim', type=int, default=256, help='the dim of the second layer in PM')



    parser.add_argument('--pm_first_dim', type=int, default=512, help='the dim of the first layer in PM')
    parser.add_argument('--pm_second_dim', type=int, default=2048, help='the dim of the second layer in PM')
    parser.add_argument('--pm_third_dim', type=int, default=256, help='the dim of the third layer in PM')
    parser.add_argument('--pt_pm_optimizer', type=str, default='RMSprop', help='The optimizer type in pretraining stage')
    #parser.add_argument('--pt_pm_optimizer', type=str, default='Adam', help='The optimizer type in pretraining stage')
    parser.add_argument('--pt_pm_momentum', type=float, default=0.9, help='value of pretraining momentum.')
    parser.add_argument('--pt_pm_weight_decay', type=float, default=0.000001, help='value of layer-wise weight decay.')
    parser.add_argument('--pt_pm_loss_patience', type=int, default=50, help='value of loss patience in pretraining')
    parser.add_argument('--pt_pm_show_patience', type=int, default=100, help='value of show patience in pretraining')
    parser.add_argument('--pt_pm_sp_weight', type=float, default=params['pt_pm_sp_weight'], help='weight of structure preservation loss')#0.001
    parser.add_argument('--sm_first_dim', type=int, default=256, help='the dim of the first layer in SM')
    parser.add_argument('--sm_second_dim', type=int, default=64, help='the dim of the second layer in SM')
    parser.add_argument('--sm_third_dim', type=int, default=256, help='the dim of the third layer in SM')
    parser.add_argument('--ft_optimizer', type=str, default='RMSprop', help='The optimizer type in pretraining stage')
    #parser.add_argument('--ft_optimizer', type=str, default='Adam', help='The optimizer type in pretraining stage')
    parser.add_argument('--ft_momentum', type=float, default=0.9, help='value of pretraining momentum.')
    parser.add_argument('--ft_weight_decay', type=float, default=0.0001, help='value of layer-wise weight decay.')
    parser.add_argument('--ft_show_patience', type=int, default=100, help='value of show patience in pretraining')
    
    args = parser.parse_args()

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True




    setup_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    
    def guide_Cluster_ability(X, labels_pred):
        """
        Compute clustering evaluation metrics.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data for internal indices.
        labels_true : array-like, shape (n_samples,)
            Ground truth labels for external metrics.
        labels_pred : array-like, shape (n_samples,)
            Predicted cluster labels.

        Returns
        -------
        dict
            Dictionary containing:
            - 'Silhouette', 'Calinski_Harabasz', 'Davies_Bouldin', 'Dunn'
        """
        # External metrics
        labels_pred = np.asarray(labels_pred).flatten()
        # Adjust labels to zero-based
        labels_pred -= labels_pred.min()


        unique_labels = np.unique(labels_pred)
        if len(unique_labels) < 2:
            # Not enough clusters to compute the scores
            return 0, 100
        else:
            # Internal indices
            #Sil = silhouette_score(X.cpu().detach().numpy(), labels_pred)
            CH = calinski_harabasz_score(X.cpu().detach().numpy(), labels_pred)
            DB = davies_bouldin_score(X.cpu().detach().numpy(), labels_pred)
            #Dunn = dunn_index(X.cpu().detach().numpy(), labels_pred)
        
        
            return CH, DB
            #Sil, , Dunn

    

    def finetune(y_pred_b, weights, best_nmi, model, feature_list, adj_hat_list, adj_hat, adj_wave, norm, weight_tensor,
                 optimizer_type,
                 learning_rate,final_lr , weight_decay, momentum, num_epochs, sp_weight, show_patience, update_interval,
                 cl_weight, labels,pl_weight , out_put_dim , cli_weight = 0.5 ):
    
        print("###################### Finetune the whole HACH-MVC model ######################")
        model.cuda()
        for i in range(model.num_views):
            feature_list[i] = feature_list[i].cuda()
            adj_hat_list[i] = adj_hat_list[i].to_dense().cuda()
            adj_wave[i] = adj_wave[i].cuda()


        # construct the optimizer and reconstructed loss function
        if optimizer_type == "RMSprop":
            optimizer = RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        else:
            optimizer = RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        loss_function = nn.MSELoss()

        model.eval()
        _, hidden, _, _, _ ,_ ,_ ,_ ,_  = model(feature_list, adj_hat_list)
        embedding = hidden.detach().cpu().cpu()
        kmeans = KMeans(n_clusters=len(np.unique(labels)), n_init=5)  # n_jobs=8
        y_pred_pr = kmeans.fit_predict(embedding)
        y_pred_pr_0 = y_pred_pr
        ACC_pr, NMI_pr, Purity, ARI, P, R, F1 = get_evaluation_results(labels.numpy(), y_pred_pr)
        print("kmeans ACC score={:.4f}".format(ACC_pr))
        print("kmeans NMI score={:.4f}".format(NMI_pr))

        Calinski_Harabasz_pr, Davies_Bouldin_pr = guide_Cluster_ability(embedding, y_pred_pr)
        print("kmeans Davies Bouldin score={:.4f}".format(Davies_Bouldin_pr))
        print("kmeans Calinski Harabasz score={:.4f}".format(Calinski_Harabasz_pr))

        #y_pred_last = y_pred_pr_hi
        y_pred_last = y_pred_pr
        '''
        ACC_pr = 0
        y_pred_pr = None
        y_pred_last = None
        '''
        DB_fi = 100
        
        
        alpha = 0.01
        
        h_acc_pr = 0

        # convergence
        loss_history = []
        
        loss_lp_history =[]
        loss_kmeans_history =[]
        loss_ct_history =[]
        loss_cli_history=[]
        loss_kl_history =[]
        
        loss_gr_history =[]
        loss_fr_history =[]
        
        acc_history = []
        nmi_history = []
        db_history = []
        eval_epochs = [] # To store the epoch number when ACC/NMI are evaluated
        first_nmi_pr_update_epoch = None # <<< Variable to track the first update
        transitioning = 0
        transitioning_ = 0
        best_nmi_ = 0


        for epoch in range(num_epochs):
            model.train()

            if epoch % update_interval == 0:
                hidden, hidden_,  tmp_q, _, _ ,adj_bar_list , _ ,adj_fusion ,adj_bar_fusion = model(feature_list, adj_hat_list)
                
                #adj_bar_tensor = torch.stack(adj_bar_list)  # or torch.cat(adj_hat_list) if appropriate
                #adj_sum = adj_hat_tensor.sum(dim=0) /len(adj_hat_list)
                #adj_bar_sum = adj_bar_tensor.sum(dim=0) /len(adj_bar_tensor)
                #_, tmp_q, _, _ = model(feature_list, adj_hat_list)
               
                    
                # update target distribution p
                tmp_q = tmp_q.data
                p = target_distribution(tmp_q)

                # evaluate clustering performance
                y_pred = tmp_q.cpu().numpy().argmax(1)
                
                
                
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                if(args.dataset_name == 'CiteSeer' or args.dataset_name == 'NUS' or args.dataset_name == 'UCI' 
                  or args.dataset_name == 'Caltech101-20' or args.dataset_name == '100leaves'  or args.dataset_name == 'Mfeat'):
                    NMI, ACC, Purity, ARI, P, R, F1 = get_evaluation_results(labels.numpy(), y_pred)
                else:
                    ACC, NMI, Purity, ARI, P, R, F1 = get_evaluation_results(labels.numpy(), y_pred)
                Calinski_Harabasz, Davies_Bouldin = guide_Cluster_ability(hidden, y_pred)
                db_history.append(Davies_Bouldin)
                # convergence--- Store metrics for plotting ---
                #acc_history.append(ACC*100)
                #nmi_history.append(NMI*100)
                eval_epochs.append(epoch) # Store the epoch number for this evaluation
                
                    
                if Davies_Bouldin <  DB_fi :
                    DB_fi = Davies_Bouldin
                    #embedding = hidden.detach().cpu().cpu()
                    if transitioning == 1 :
                            influential_nodes , most_influentials = update_influential_nodes(adj_bar_fusion.cpu(),db_history,loss_cli_history,epoch)
                            #influential_nodes , most_influentials = update_influential_nodes(updated_adj,db_history,loss_cli_history,epoch)
                    elif epoch % 10 or epoch == 0 :
                        influential_nodes , most_influentials = update_influential_nodes(adj_bar_fusion.cpu(),db_history,loss_cli_history,epoch)
                elif Davies_Bouldin >= DB_fi and DB_fi == 100 :
                    influential_nodes , most_influentials = update_influential_nodes(adj_bar_fusion.cpu(),db_history,loss_cli_history,epoch)
                   
                
                
            hidden, hidden_, q, X_bar_list,_ ,A_bar_list , hidden_list ,_, _ = model(feature_list, adj_hat_list)
            #hidden,q, X_bar_list, A_bar_list = model(feature_list, adj_hat_list)
            #ACC, NMI, Purity, ARI, P, R, F1 = get_evaluation_results(labels.numpy(), y_pred)
            
            if NMI > best_nmi_  :
                best_nmi_ = NMI
                
                
                if NMI > best_nmi:
                    best_nmi = NMI
                    weights = deepcopy(model.state_dict())
                    
                
                
                #if epoch > 50 :
                print('Number of Unique Influential Nodes:', len(most_influentials))
                #visualize_clusters(torch.tensor(hidden), q.data.cpu().numpy().argmax(1).squeeze() , "Mfeat")
                #print('Iter {}'.format(epoch), ':ACC {:.4f}'.format(ACC),':NMI {:.4f}'.format(NMI), ', f1 {:.4f}'.format(F1), ', Davies_Bouldin {:.4f}'.format(Davies_Bouldin) )
                
                y_pred_b = y_pred
                    
                
                ACC, NMI, Purity, ARI, P, R, F1 = get_evaluation_results(labels.numpy(), y_pred_b)
                print('Iter {}'.format(epoch), ':ACC {:.4f}'.format(ACC),':NMI {:.4f}'.format(NMI), ', f1 {:.4f}'.format(F1), ', Davies_Bouldin {:.4f}'.format(Davies_Bouldin) , 'Calinski_Harabasz {:.4f}'.format(Calinski_Harabasz))
                
                nni.report_intermediate_result(0)


            
            if transitioning == 1 :
                transitioning = 0
                '''
                if transitioning_ == 0:
                    transitioning_ = 1
                '''
                if y_pred_b.any() :
                    y_pred_pr = y_pred_b
                else:
                    y_pred_pr = y_pred_
                    
                Davies_Bouldin_pr = Davies_Bouldin
                h_acc_pr = 1
                
                if first_nmi_pr_update_epoch is None:
                    first_nmi_pr_update_epoch = eval_epochs[-1] # Store the EVAL epoch
            
            dataset_name = args.dataset_name
                
            optimizer.zero_grad()
            lossr_list = []
            losss_list = []
            if(args.ft_h < 2 and args.ft_k <3  and args.ft_ch == 1):
                args.ft_ch = 0
                args.ft_sp_weight = args.ft_sp_weight * 2
                args.ft_cli_weight = args.ft_cli_weight / 2
                args.ft_pl_weight = args.ft_pl_weight * 2
                
            elif(args.ft_h > 3 and args.ft_k > 3 and args.ft_ch == 1):
                args.ft_ch = 0
                args.ft_sp_weight = args.ft_sp_weight / 2
                args.ft_cli_weight = args.ft_cli_weight * 2
                args.ft_pl_weight = args.ft_pl_weight / 2
                
            #contrastive_losses_local = []
            for v in range(model.num_views):
                #print("A_bar_list[v] : " , A_bar_list[v])                             
                #print("adj_wave[v] : " , adj_wave[v])    
                lossr_list.append(sp_weight * loss_function(feature_list[v], X_bar_list[v]))
                losss_list.append(
                    #sp_weight * F.binary_cross_entropy(A_bar_list[v].view(-1), adj_wave[v].to_dense().view(-1)))
                    (1-((2*epoch)/(3*num_epochs))) * F.binary_cross_entropy(A_bar_list[v].view(-1), adj_wave[v].to_dense().view(-1)))
                    #(1-(epoch/num_epochs))* F.binary_cross_entropy(A_bar_list[v].view(-1), adj_wave[v].to_dense().view(-1)))



            loss_lr = sum(lossr_list)
            loss_lr_value = float(loss_lr.item())
            loss_fr_history.append(loss_lr_value)
            loss_ls = sum(losss_list)
            loss_ls_value = float(loss_ls.item())
            loss_gr_history.append(loss_ls_value)
            #print('len influential_nodes' , len(influential_nodes))
            #-(epoch/num_epochs)
            #loss_cli = cli_weight * contrastive_loss_influential(hidden, influential_nodes , most_influentials )
            
            
            loss_cli_a =  contrastive_loss_influential(hidden, influential_nodes , most_influentials , num_epochs , epoch)
            
            loss_cli = cli_weight *((1+(epoch))/(num_epochs)) * loss_cli_a
            loss_cli_value = float(loss_cli_a.item())
            loss_cli_history.append(loss_cli_value)
            #if loss_cli_value <= 0.1 : 
                
            #    loss_ct_history.append(loss_cli_value)
            #else :
            #    loss_ct_history.append( 0.1 * loss_cli_value/loss_cli_value)
            
            loss_lc_a = F.kl_div(q, p)

            #loss_lc_a = torch.mean(torch.sum(torch.abs(p - q.log()), dim=1))
            #loss_lc_a = torch.mean(torch.sum(torch.abs(p - q), dim=1))
            loss_lc = cl_weight * loss_lc_a
            loss_lc_value = float(loss_lc_a.item())
            loss_kl_history.append(loss_lc_value)
            #loss_lc = (1-(epoch/num_epochs)) * F.kl_div(q.log(), p)
            loss_func  = nn.CrossEntropyLoss(reduction="none")
            #loss_lp = pl_weight * loss_func(q, torch.tensor(y_pred_pr).long().to(device))     #loss function
            #loss_lp = pl_weight * ((1-(epoch/num_epochs))*(loss_func(q, torch.tensor(y_pred_pr).long().to(device)) ) ) 
            
            loss_lp_a = loss_func(q, torch.tensor(y_pred_pr).long().to(device))
            loss_lp_k = loss_func(q, torch.tensor(y_pred_pr_0).long().to(device))
            loss_lp_k = torch.mean(loss_lp_k)
            loss_lp_b = torch.mean(loss_lp_a)

            loss_lp = pl_weight * (1-((2*epoch)/(3*num_epochs)))*(loss_lp_a )
            
            loss_lp = torch.mean(loss_lp)
            loss_lp_value = float(loss_lp_b.item())
            loss_lp_value_k = float(loss_lp_k.item())
            loss_lp_history.append(loss_lp_value)
            loss_kmeans_history.append(loss_lp_value_k)
            #loss_ic = vectorized_contrastive_learning(adj_fusion , influential_nodes, hidden, cutoff=1, temperature=0.7)
            #loss_contrastive_local = alpha * sum(contrastive_losses_local) / len(contrastive_losses_local)
            
            loss = loss_ls + loss_lr + loss_lc + loss_lp + loss_cli

                
            # + loss_contrastive_local

            loss.backward()
            optimizer.step(closure=None)
            loss_value = float(loss.item())
            
            # convergence --- Store Loss History ---
            loss_history.append(loss_value)
            
            #loss_lp_value = float(loss_lp_b.item())
            #loss_lp_history.append(loss_lp_value)
            arr_lp = np.array(loss_lp_history)
            m_lp = arr_lp.mean()
            s_lp = arr_lp.std()
            lp = loss_lp_value - m_lp
            lp_md = m_lp - 2 *s_lp
            
            if len(loss_lp_history) > num_epochs/3.3 and  lp < 0 and Davies_Bouldin < Davies_Bouldin_pr:
                #y_pred_ = y_pred
                y_pred_ = y_pred_pr
                '''
                print( ' lp : ', lp)
                print( ' lp_md : ', lp_md)
                print( ' loss_lp_value : ', loss_lp_value)
                print( ' loss_lp_history.mean() : ', m_lp)
                print( ' loss_lp_history.std() : ', s_lp)
                '''
                transitioning = 1
                
            #if (epoch + 1) % show_patience == 0:
            #    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss_value))

            #new_lr = (learning_rate) - epoch * ((learning_rate) - final_lr) / num_epochs #linear decacy
            #for param_group in optimizer.param_groups:
            #    param_group['lr'] = new_lr  # Update the learning rate in the optimiser

        #################saving the node representation
        model.eval()
        model.load_state_dict(weights)
        hidden, hidden_ , q, X_bar_list, _ ,A_bar_list,_ ,_,_ = model(feature_list, adj_hat_list)



        

        return q.data.cpu().numpy().argmax(1), weights, best_nmi , hidden


    #  del_files("./data/adj_matrix/")
    del_files("./data/ec_feature/")
    del_files("./data/graph_new_weight/")
    del_files("./data/pt_weight/")


    y_pred_b = np.array([0])
    
    sd = 0
    pre_train = False
    best_ft_weight = None
    #noises = [0.0 , 0.005 , 0.1 , 0.15 , 0.2 , 0.25 , 0.3]
    noises = [0.0]
    for fnoise in noises :
        args = parser.parse_args()
        print(f'----------------------------- feature noise: {fnoise} ---------------------------------------- ' )
        all_ACC = []
        all_NMI = []
        all_Purity = []
        all_ARI = []
        all_F = []
        all_P = []
        all_R = []
        all_PT_TIME = []
        all_FT_TIME = []
        all_DavB = []
        all_CH = []
        best_nmi = 0
        args.pm_feature_noise_level = fnoise
        args.pm_struct_noise_level = 0
        default_ft_dahk = 0
        if(args.ft_dahk == 1 ):
            default_ft_dahk = 1
        print('args.pm_struct_noise_level : ' , args.pm_struct_noise_level )
        print('args.pm_feature_noise_level : ' , args.pm_feature_noise_level )
        for i in range(args.n_repeated):
            
            args.ft_dahk = default_ft_dahk
            print(" --------------------------- run num:----------------------- " , i+1 )
            if(args.dataset_name == 'WebKB'):
                args.pm_knns = 25
                args.pt_pm_sp_weight = 0.0000001
                args.ft_num_epochs = 451 
                #args.ft_num_epochs = 801 
                args.pt_pm_num_epochs = 800
                args.ft_cl_weight = 0.01

                if(args.ft_dahk == 1 ):
                    
                    if(args.pm_feature_noise_level > 0.1):
                        args.pt_pm_sp_weight = args.pt_pm_sp_weight + (args.pt_pm_sp_weight * (args.pm_feature_noise_level * (10 ** (args.pm_feature_noise_level * 10)))) 
                    else:
                        args.ft_dahk = 0
            if(args.dataset_name == 'MSRC_v1'):
                args.pm_knns = 8
                args.pt_pm_sp_weight = 0.1
                args.ft_num_epochs = 401 
                args.pt_pm_num_epochs = 800
                args.ft_cl_weight = 1.0 
                if(args.ft_dahk == 1 ):
                    if(args.pm_feature_noise_level < 0.2):
                        args.ft_dahk = 0
                        args.pt_pm_sp_weight = args.pt_pm_sp_weight + (args.pm_struct_noise_level * 10) + (args.pm_feature_noise_level * (10 ** ((args.pm_feature_noise_level * 10))))
                    else:
                        args.pt_pm_sp_weight = args.pt_pm_sp_weight + (args.pt_pm_sp_weight * (args.pm_feature_noise_level * (10 ** (args.pm_feature_noise_level * 10))))
            
            if(args.dataset_name == 'bbcsport'):
                args.pm_knns = 25
                args.pt_pm_sp_weight = 0.01
                args.ft_num_epochs = 801 
                args.pt_pm_num_epochs = 800
                args.ft_cl_weight = 0.000001
                if(args.pm_feature_noise_level < 0.05):
                    args.ft_dahk = 0
                if(args.ft_dahk == 1 ):
                    if(args.pm_feature_noise_level < 0.15 and args.pm_feature_noise_level > 2 ):
                        args.pt_pm_sp_weight = args.pt_pm_sp_weight + (args.pt_pm_sp_weight * (args.pm_feature_noise_level * (10 ** (args.pm_feature_noise_level * 10)))) 
                        

                                 
            if(args.dataset_name == '100leaves'):
                args.pm_knns = 16
                args.pt_pm_sp_weight = 0.1
                args.ft_num_epochs = 851 
                args.pt_pm_num_epochs = 800
                args.ft_cl_weight = 1.0
                args.ft_pl_weight = 100000.0
                
                if(args.ft_dahk == 1 ):
                    args.pt_pm_sp_weight = args.pt_pm_sp_weight + (args.pm_struct_noise_level * 10) + (args.pm_feature_noise_level * (10 ** ((args.pm_feature_noise_level * 10)+1)))
                
            if(args.dataset_name == 'Mfeat'):
                args.pm_knns = 25
                args.pt_pm_sp_weight = 0.01
                args.ft_num_epochs = 351 
                args.pt_pm_num_epochs = 800
                args.ft_cl_weight = 0.1
                if(args.ft_dahk == 1 ):
                    if(args.pm_feature_noise_level < 0.1):
                        args.ft_dahk = 0
                
            if(args.dataset_name == 'UCI'):
                args.pm_knns = 25
                args.pt_pm_sp_weight = 0.01
                args.ft_num_epochs = 801 
                args.pt_pm_num_epochs = 800
                args.ft_cl_weight = 0.01
                args.ft_pl_weight = 10000.0
                if(args.ft_dahk == 1 ):
                    if(args.pm_feature_noise_level < 0.1):
                        args.ft_dahk = 0
                
            if(args.dataset_name == 'Caltech101-20'):
                args.pm_knns = 8
                args.pt_pm_sp_weight = 0.01
                args.ft_num_epochs = 701 
                args.pt_pm_num_epochs = 800
                args.ft_cl_weight = 10.0
                args.ft_pl_weight = 10000.0
                
                if(args.ft_dahk == 1 ):
                    if(args.pm_feature_noise_level < 0.1):
                        args.pt_pm_sp_weight = args.pt_pm_sp_weight + (args.pt_pm_sp_weight * (args.pm_feature_noise_level * (10 ** (args.pm_feature_noise_level * 10)))) 
                    else:
                        args.pt_pm_sp_weight = args.pt_pm_sp_weight + (args.pt_pm_sp_weight * (args.pm_feature_noise_level * (10 ** (args.pm_feature_noise_level * 10)-1))) 
            
            if(args.dataset_name == 'CiteSeer'):
                args.pm_knns = 25
                args.pt_pm_sp_weight = 10.0
                args.ft_num_epochs = 1401 
                args.pt_pm_num_epochs = 600
                args.ft_cl_weight = 1.0
                args.ft_pl_weight = 1000.0
                
                if(args.ft_dahk == 1 ):
                    args.pt_pm_sp_weight = args.pt_pm_sp_weight + (args.pt_pm_sp_weight * (args.pm_feature_noise_level * (10 ** (args.pm_feature_noise_level * 10)))) 


            if (args.pm_struct_noise_level > 0.3):
                args.pm_knns = 25

            print('args.pm_knns: ' , args.pm_knns)
            print('args.pm_struct_noise_level : ' , args.pm_struct_noise_level )
            adj_list_new = pretraining(pre_train, args)
            
            labels, feature_list, adj_wave_list, adj_hat_list, norm_list, weight_tensor_list = load_data_clusterft(adj_list_new,
                direction_path=args.direction, dataset_name=args.dataset_name,
                load_saved = True, k_nearest_neighobrs=args.pm_knns , struct_noise_level = args.pm_struct_noise_level,
                feature_noise_level = args.pm_feature_noise_level)

            args.num_classes = len(np.unique(labels))
            args.num_views = len(feature_list)
            view_dims = []
            for j in range(args.num_views):
                view_dims.append(feature_list[j].shape[1])
            print('view_dims',view_dims)
            pm_hidden_dims = [args.pm_first_dim, args.pm_second_dim, args.pm_third_dim]
            am_hidden_dims = [args.am_first_dim, args.am_second_dim]
            sm_hidden_dims = [args.sm_first_dim, args.sm_second_dim, args.sm_third_dim]

            # pretraining or loading the weights of HACH-MVC
            ec_feat_save_direction = './data/ec_feature/'
            if not os.path.exists(ec_feat_save_direction):
                os.makedirs(ec_feat_save_direction)
            ec_feat_save_path = ec_feat_save_direction + args.dataset_name + '.npy'
            pt_weight_save_direction = './data/pt_weight/'
            if not os.path.exists(pt_weight_save_direction):
                os.makedirs(pt_weight_save_direction)
            pt_weight_save_path = pt_weight_save_direction +args.dataset_name + '.pkl'
            ft_weight_save_direction = './data/ft_weight/'+args.dataset_name + '.pkl'

            # exit()
            print("############### loading the pretrained wieghts.... ###############")
            model = HACHMVC(view_dims, pm_hidden_dims,am_hidden_dims, sm_hidden_dims, args.num_classes)

            model.load_state_dict(torch.load(pt_weight_save_path))

            # finetune the whole HACH-MVC model
            ft_begin_time = time.time()#input adj adj_hat adj_wave is numpy
            ##best_ft_weight Global variables
            predicted, best_ft_weight, best_val_nmi , hidden  = finetune(y_pred_b, best_ft_weight, best_nmi, model=model, feature_list=feature_list, adj_hat_list=adj_hat_list,
                             adj_hat=adj_hat_list[0], adj_wave=adj_wave_list, norm=norm_list[0], weight_tensor=weight_tensor_list[0],
                             optimizer_type=args.ft_optimizer, learning_rate=args.ft_lr, final_lr = args.final_lr, momentum=args.ft_momentum,
                             weight_decay=args.ft_weight_decay, num_epochs=args.ft_num_epochs, sp_weight=args.ft_sp_weight,
                             show_patience=args.ft_show_patience, update_interval=args.ft_update_interval,
                             cl_weight=args.ft_cl_weight, labels=labels,pl_weight=args.ft_pl_weight, out_put_dim = args.pm_third_dim , cli_weight = args.ft_cli_weight)
            if best_val_nmi > best_nmi:
                best_nmi = best_val_nmi
            ft_cost_time = time.time() - ft_begin_time
            ACC, NMI, Purity, ARI, P, R, F1 = get_evaluation_results(labels.numpy(), predicted)
            print('ACC, NMI, Purity, ARI, P, R, F1',ACC, NMI, Purity, ARI, P, R, F1)
            Calinski_Harabasz, Davies_Bouldin = guide_Cluster_ability(hidden, predicted)
            # nni.report_final_result(ACC)
            ret1 = [ACC, NMI, Purity, ARI, P, R, F1,Calinski_Harabasz ,Davies_Bouldin,i+1]
            

            # pred_save_path = './data/pred/' + args.dataset_name + '.mat'
            # sio.savemat(pred_save_path, {'pred': predicted})

            all_ACC.append(ACC)
            all_NMI.append(NMI)
            all_Purity.append(Purity)
            all_ARI.append(ARI)
            all_P.append(P)
            all_R.append(R)
            all_F.append(F1)
            
            all_DavB.append(Davies_Bouldin)
            all_CH.append(Calinski_Harabasz)
            
            all_FT_TIME.append(ft_cost_time)
        nni.report_final_result(ACC)
        # append result to .txt file
        fp = open("results.txt", "a+", encoding="utf-8")
        # fp = open("results_" + args.dataset_name + ".txt", "a+", encoding="utf-8")
        fp.write("dataset_name: {}\n\n".format(args.dataset_name))
        fp.write("structral noise: {}\n".format(args.pm_struct_noise_level))
        fp.write("feature noise: {}\n".format(args.pm_feature_noise_level))
        #fp.write("ft_sp_weight: {}\n".format(args.ft_pl_weight))
        #fp.write("ft_cl_weight: {}\n".format(args.ft_cl_weight))
        fp.write("h hop: {}\n".format(args.ft_h))
        fp.write("topK: {}\n".format(args.ft_k))
        fp.write("dynamically adjaust h & k : {}\n".format(default_ft_dahk))
        fp.write("ACC: {:.2f}\t{:.2f}\n".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
        fp.write("NMI: {:.2f}\t{:.2f}\n".format(np.mean(all_NMI) * 100, np.std(all_NMI) * 100))
        fp.write("Purity: {:.2f}\t{:.2f}\n".format(np.mean(all_Purity) * 100, np.std(all_Purity) * 100))
        fp.write("ARI: {:.2f}\t{:.2f}\n".format(np.mean(all_ARI) * 100, np.std(all_ARI) * 100))
        fp.write("P: {:.2f}\t{:.2f}\n".format(np.mean(all_P) * 100, np.std(all_P) * 100))
        fp.write("R: {:.2f}\t{:.2f}\n".format(np.mean(all_R) * 100, np.std(all_R) * 100))
        fp.write("F: {:.2f}\t{:.2f}\n".format(np.mean(all_F) * 100, np.std(all_F) * 100))
        fp.write("Davies_Bouldin: {:.2f}\t{:.2f}\n".format(np.mean(all_DavB) , np.std(all_DavB) ))
        fp.write("Calinski_Harabasz: {:.2f}\t{:.2f}\n".format(np.mean(all_CH) , np.std(all_CH) ))
        # fp.write("Pretrain Time: {:.2f}\t{:.2f}\n".format(np.mean(all_PT_TIME), np.std(all_PT_TIME)))
        fp.write("Finetune Time: {:.2f}\t{:.2f}\n\n".format(np.mean(all_FT_TIME), np.std(all_FT_TIME)))
        fp.close()

        