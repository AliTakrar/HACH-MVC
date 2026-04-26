'''
@Project: HACH-MVC
@Author : Ali Takrar
@Email  : takrar.co@gmail.com
@File   : ops_ev
@Desc
    Evaluation module for clustering
'''
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster._supervised import check_clusterings
from scipy.optimize import linear_sum_assignment as linear_assignment

def get_evaluation_results(labels_true, labels_pred):
    """
    :param y_true:
        data type: numpy.ndarray
        shape: (n_samples,)
        sample: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
    :param y_pred:
        data type: numpy.ndarray
        shape: (n_samples,)
        sample: [ 0  1  2  3  4  5  6  7  8  9ops_io.py 10 11 12 13 14 15 16 17 18 19]
    :return: ACC, NMI, Purity, ARI
    """
    NMI = normalized_mutual_info_score(labels_true, labels_pred)
    ARI = adjusted_rand_score(labels_true, labels_pred)
    ACC = clustering_accuracy(labels_true, labels_pred)
    P, R, F = b3_precision_recall_fscore(labels_true, labels_pred)
    Purity = clustering_purity(labels_true.reshape((-1, 1)), labels_pred.reshape(-1, 1))

    return ACC, NMI, Purity, ARI, P, R, F







def clustering_purity(labels_true, labels_pred):
    """
    :param labels_true: Ground truth labels (numpy.ndarray), shape: (n_samples,)
    :param labels_pred: Predicted cluster labels (numpy.ndarray), shape: (n_samples,)
    :return: Purity (float)
    """
    # Ensure labels are numpy arrays
    labels_true = np.asarray(labels_true).flatten()
    labels_pred = np.asarray(labels_pred).flatten()

    # Map labels to zero-based indexing if not already
    labels_true = labels_true - np.min(labels_true)
    labels_pred = labels_pred - np.min(labels_pred)

    # Create contingency matrix
    max_true_label =  int(np.max(labels_true) + 1)
    max_pred_label =  int(np.max(labels_pred) + 1)
    contingency_matrix = np.zeros((max_true_label, max_pred_label), dtype=int)

    for true_label, pred_label in zip(labels_true, labels_pred):
        contingency_matrix[int(true_label), int(pred_label)] += 1

    # Calculate purity using the max in each cluster
    max_in_cluster = np.sum(np.amax(contingency_matrix, axis=0))
    purity = max_in_cluster / len(labels_true)

    return purity



from scipy.optimize import linear_sum_assignment

def clustering_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy.

    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy: clustering accuracy, in [0, 1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size

    # Construct the contingency matrix
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # Solve the optimal assignment problem
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    # Compute accuracy based on the optimal assignment
    accuracy = sum([w[i, j] for i, j in zip(row_ind, col_ind)]) / y_pred.size
    return accuracy


def b3_precision_recall_fscore(labels_true, labels_pred):
    """Compute the B^3 variant of precision, recall and F-score.
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.
    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f_score: calculated f_score
    Reference
    ---------
    Amigo, Enrique, et al. "A comparison of extrinsic clustering evaluation
    metrics based on formal constraints." Information retrieval 12.4
    (2009): 461-486.
    """
    # Check that labels_* are 1d arrays and have the same size

    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    # Check that input given is not the empty set
    if labels_true.shape == (0,):
        raise ValueError(
            "input labels must not be empty.")

    # Compute P/R/F scores
    n_samples = len(labels_true)
    true_clusters = {}  # true cluster_id => set of sample indices
    pred_clusters = {}  # pred cluster_id => set of sample indices

    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0

    intersections = {}

    for i in range(n_samples):
        pred_cluster_i = pred_clusters[labels_pred[i]]
        true_cluster_i = true_clusters[labels_true[i]]

        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection

        precision += len(intersection) / len(pred_cluster_i)
        recall += len(intersection) / len(true_cluster_i)

    precision /= n_samples
    recall /= n_samples

    f_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f_score
