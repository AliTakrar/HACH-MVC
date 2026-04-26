'''
@Project: HACH-MVC
@Author : Ali Takrar
@Email  : takrar.co@gmail.com
@File   : ops_io

@Desc
    Functions of i/o operations
'''

import os
import pdb
import time
import torch
import openpyxl

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale, maxabs_scale, normalize, robust_scale, scale


def perturb_graph_structure(adj, noise_ratio):
    """
    Randomly removes edges from the adjacency matrix to simulate structural noise.
    adj: scipy sparse matrix
    noise_ratio: float (0.0 to 1.0), percentage of edges to drop
    """
    if noise_ratio <= 0:
        return adj
    

    # Convert to COO format to access edges easily
    adj_coo = adj.tocoo()
    
    # Get total number of edges
    num_edges = adj_coo.nnz
    num_keep = int(num_edges * (1 - noise_ratio))
    
    # Randomly select indices of edges to keep
    #keep_indices = np.random.choice(num_edges, num_keep, replace=False)
    keep_indices = np.random.choice(num_edges, num_keep, replace=False)
    
    # Construct the new perturbed matrix
    new_row = adj_coo.row[keep_indices]
    new_col = adj_coo.col[keep_indices]
    new_data = adj_coo.data[keep_indices]
    
    new_adj = sp.csc_matrix((new_data, (new_row, new_col)), shape=adj.shape)
    

    return new_adj
import numpy as np



def add_gaussian_noise(data, sigma=0.1):
    """
    Adds Gaussian noise to the input data.
    
    Parameters:
    - data: Input feature vectors (numpy array).
    - sigma (float): Standard deviation of the noise (noise level).

    
    Returns:
    - noisy_data: Data with added Gaussian noise.
    """
    # Ensure data is a numpy array
    data = np.array(data)
    import random

    
    # Generate noise: N(0, sigma^2)
    # loc=0 is Mean (mu), scale=sigma is Standard Deviation
    noise = np.random.normal(loc=0, scale=sigma, size=data.shape)
    
    noisy_data = data + noise

    return noisy_data
def add_masking_noise(data, noise_ratio=0.1):
    """
    Randomly masks (sets to 0) features to simulate missing words/text noise.
    Safe for sparse text data like WebKB, BBCSport, CiteSeer.
    """
    if noise_ratio <= 0:
        return data
        
    rng = np.random
    
    # Create a mask of the same shape as data
    # True means "drop this feature"
    mask = rng.random(data.shape) < noise_ratio
    
    # Deep copy to avoid modifying original
    noisy_data = data.copy()
    
    # Set masked values to 0
    noisy_data[mask] = 0
    
    return noisy_data

def load_data_clusterft(adj_new, direction_path, dataset_name, load_saved=False,
              k_nearest_neighobrs=50, struct_noise_level = 0.0, feature_noise_level = 0.0):

    # construct the target path and load the data
    print("Prepare to load all related data of " + dataset_name + " ........")
    target_path = direction_path + '/' + dataset_name + '.mat'
    data = sio.loadmat(target_path)
    prunning_one = True
    prunning_two = True
    common_neighbors = 2

    # construct all needed data
    try:
        if(dataset_name == 'Mfeat' or dataset_name == '100leaves' ):
            features = data['data']
            labels = data['truelabel'][0][1].flatten()
        elif(dataset_name == 'MSRC_v1'):
            features = data['fea']
            labels = data['gt'].flatten() 
        else:
            features = data['X']
            labels = data['Y'].flatten()
        feature_list = []
        adj_wave_list = []
        adj_hat_list = []
        norm_list = []
        weight_tensor_list = []
        # load data of each view
        for i in range(features.shape[1]):
            print("Loading the data of the " + str(i) + "th view .......")

            if(dataset_name == 'bbcsport' or dataset_name == 'CiteSeer' ):
                dns_features = features[0][i].toarray()
            elif(dataset_name == 'Mfeat' or dataset_name == '100leaves' ):
                dns_features = features[0][i].transpose()
            else:
                dns_features = features[0][i]
            # List of datasets that contain dense/continuous features (Image/Digit/Shape)
            # These require Gaussian Noise to simulate sensor/feature noise.
            dense_datasets = ['MSRC_v1', 'NUS', '100leaves', 'Mfeat', 'UCI', 'Caltech101-20']
            
            if feature_noise_level > 0:
                # Calculate ratio (0.1, 0.2, etc.)
                ratio = feature_noise_level
                
                if dataset_name in dense_datasets:
                    # Gaussian Noise: We scale the noise relative to the standard deviation of the features
                    # Sigma = ratio * std_dev (Standard Signal-to-Noise approach)
                    sigma = ratio * np.std(dns_features)
                    noisy_feature = add_gaussian_noise(dns_features, sigma=sigma)
                    
                    
                else:
                    # Text/Sparse Datasets (WebKB, BBCSport, CiteSeer)
                    # Masking Noise: We just pass the ratio (probability of masking a word)
                    # DO NOT multiply by std() here.
                    noisy_feature = add_masking_noise(dns_features, noise_ratio=ratio)
            
                
            else:
                noisy_feature = dns_features

            fea, wave, hat, norm, weight = load_single_view_data(noisy_feature, dataset_name, i,
                                                                 load_saved, k_nearest_neighobrs, prunning_one,
                                                                 prunning_two, common_neighbors,adj_new, struct_noise_level)
            feature_list.append(fea)
            adj_wave_list.append(wave)
            adj_hat_list.append(hat)
            norm_list.append(norm)
            weight_tensor_list.append(weight)

    except KeyError:
        print("An error is raised during loading the features....")
        exit()

    #labels = data['truelabel'][0][1].flatten()  # <class 'numpy.ndarray'> (n_samples, )
    labels = label_from_zero(labels)
    labels = torch.from_numpy(labels).float()

    return labels, feature_list, adj_wave_list, adj_hat_list, norm_list, weight_tensor_list


def load_single_view_data(feature, dataset_name, view_no, load_saved, k_nearest_neighobrs,
                          prunning_one, prunning_two, common_neighbors, adj_new, struct_noise_level=0.0):
    normalization = 'normalize'
    if normalization == 'minmax_scale':
        feature = minmax_scale(feature)
    elif normalization == 'maxabs_scale':
        feature = maxabs_scale(feature)
    elif normalization == 'normalize':
        feature = normalize(feature)
    elif normalization == 'robust_scale':
        feature = robust_scale(feature)
    elif normalization == 'scale':
        feature = scale(feature)
    elif normalization == '255':
        feature = np.divide(feature, 255.)
    elif normalization == '50':
        feature = np.divide(feature, 50.)
    elif normalization == 'no':
        pass
    else:
        print("Please enter a correct normalization type!")
        pdb.set_trace()

    save_direction = './data/adj_matrix/' + dataset_name + '/'
    if not os.path.exists(save_direction):
        os.makedirs(save_direction)
    if load_saved is not True:
        # construct three kinds of adjacency matrix
        print("Constructing the adjacency matrix of " + dataset_name + " in the " + str(view_no) + "th view ......")
        adj, adj_wave, adj_hat = construct_adjacency_matrix(prunning_one, prunning_two, common_neighbors, adj_new)
        

        # transform to sparse float tensor
        # features = construct_sparse_float_tensor(features)
        if struct_noise_level > 0:
            print(f"Applying structural noise (ratio={struct_noise_level}) to view {view_no}...")
            adj = perturb_graph_structure(adj, struct_noise_level)
            # Reconstruct derived matrices based on perturbed adj
            adj_wave = adj.copy() 
            adj_hat = construct_adjacency_hat(adj)

        # save these scale and matrix
        print("Saving the adjacency matrix to " + save_direction)
        sp.save_npz(save_direction + str(view_no) + '_adj.npz', adj)
        sp.save_npz(save_direction + str(view_no) + '_adj_wave.npz', adj_wave)
        sp.save_npz(save_direction + str(view_no) + '_adj_hat.npz', adj_hat)

        
    print("load the saved adjacency matrix of " + dataset_name)
    adj = sp.load_npz(save_direction + str(view_no) + '_adj.npz')
    adj_wave = sp.load_npz(save_direction + str(view_no) + '_adj_wave.npz')
    adj_hat = sp.load_npz(save_direction + str(view_no) + '_adj_hat.npz')

    
    # transform to sparse float tensor
    # features = construct_sparse_float_tensor(features)

    if sp.isspmatrix_csr(feature):
        feature = feature.todense()
    feature = torch.from_numpy(feature).float()
    adj_wave = construct_sparse_float_tensor(adj_wave)
    adj_hat = construct_sparse_float_tensor(adj_hat)

    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)  # <class 'float'>
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()  # <class 'numpy.float64'>
    weight_mask = adj_wave.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight  # <class 'torch.Tensor'>

    return feature, adj_wave, adj_hat, norm, weight_tensor




def label_from_zero(labels):
    min_num = min(set(labels))
    return labels - min_num


def construct_adjacency_matrix(prunning_one, prunning_two, common_neighbors, adj_new):
    start_time = time.time()

    if prunning_one:
        # Pruning strategy 1
        original_adj_wave = adj_new.A
        judges_matrix = original_adj_wave == original_adj_wave.T
        np_adj_wave = original_adj_wave * judges_matrix
        adj_wave = sp.csc_matrix(np_adj_wave)
    else:
        # transform the matrix to be symmetric (Instead of Pruning strategy 1)
        np_adj_wave = construct_symmetric_matrix(adj_new.A)
        adj_wave = sp.csc_matrix(np_adj_wave)

    # obtain the adjacency matrix without self-connection
    adj = sp.csc_matrix(np_adj_wave)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    if prunning_two:
        # Pruning strategy 2
        adj = adj.A
        b = np.nonzero(adj)
        rows = b[0]
        cols = b[1]
        dic = {}
        for row, col in zip(rows, cols):
            if row in dic.keys():
                dic[row].append(col)
            else:
                dic[row] = []
                dic[row].append(col)
        for row, col in zip(rows, cols):
            if len(set(dic[row]) & set(dic[col])) < common_neighbors:
                adj[row][col] = 0
        adj = sp.csc_matrix(adj)
        adj.eliminate_zeros()

    # construct the adjacency hat matrix
    adj_hat = construct_adjacency_hat(adj)  # <class 'scipy.sparse.coo.coo_matrix'>  (n_samples, n_samples)
    print("The construction of adjacency matrix is finished!")
    print("The time cost of construction: ", time.time() - start_time)

    return adj, adj_wave, adj_hat


def construct_adjacency_hat(adj):
    """
    :param adj: original adjacency matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))  # <class 'numpy.ndarray'> (n_samples, 1)
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    # <class 'scipy.sparse.coo.coo_matrix'>  (n_samples, n_samples)
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized


def construct_symmetric_matrix(original_matrix):
    """
        transform a matrix (n*n) to be symmetric
    :param np_matrix: <class 'numpy.ndarray'>
    :return: result_matrix: <class 'numpy.ndarray'>
    """
    result_matrix = np.zeros(original_matrix.shape, dtype=float)
    num = original_matrix.shape[0]
    for i in range(num):
        for j in range(num):
            if original_matrix[i][j] == 0:
                continue
            elif original_matrix[i][j] == 1:
                result_matrix[i][j] = 1
                result_matrix[j][i] = 1
            else:
                print("The value in the original matrix is illegal!")
                pdb.set_trace()
    assert (result_matrix == result_matrix.T).all() == True

    if ~(np.sum(result_matrix, axis=1) > 1).all():
        print("There existing a outlier!")
        pdb.set_trace()

    return result_matrix


def construct_sparse_float_tensor(np_matrix):
    """
        construct a sparse float tensor according a numpy matrix
    :param np_matrix: <class 'numpy.ndarray'>
    :return: torch.sparse.FloatTensor
    """
    sp_matrix = sp.csc_matrix(np_matrix)
    three_tuple = sparse_to_tuple(sp_matrix)
    sparse_tensor = torch.sparse.FloatTensor(torch.LongTensor(three_tuple[0].T),
                                             torch.FloatTensor(three_tuple[1]),
                                             torch.Size(three_tuple[2]))
    return sparse_tensor


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()

    # sparse_mx.row/sparse_mx.col  <class 'numpy.ndarray'> [   0    0    0 ... 2687 2694 2706]
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()  # <class 'numpy.ndarray'> (n_edges, 2)
    values = sparse_mx.data  # <class 'numpy.ndarray'> (n_edges,) [1 1 1 ... 1 1 1]
    shape = sparse_mx.shape  # <class 'tuple'>  (n_samples, n_samples)
    return coords, values, shape


