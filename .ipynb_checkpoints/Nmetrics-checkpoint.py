import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, v_measure_score, accuracy_score
from scipy.optimize import linear_sum_assignment
from collections import Counter

nmi = normalized_mutual_info_score
vmeasure = v_measure_score
ari = adjusted_rand_score

def acc(y_true, y_pred):
    """
    Compute clustering accuracy, handling label alignment.
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    ind = np.column_stack((row_ind, col_ind))
    
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def purity(y_true, y_pred):
    """
    Compute clustering purity.
    """
    total_correct = 0
    unique_clusters = np.unique(y_pred)
    
    for cluster in unique_clusters:
        # Get the true labels of all samples in the current cluster
        labels_in_cluster = y_true[y_pred == cluster]
        
        # Skip if the cluster has no samples
        if len(labels_in_cluster) == 0:
            continue
            
        # Find the most common true label in the current cluster
        counts = Counter(labels_in_cluster)
        most_common_label, count = counts.most_common(1)[0]
        total_correct += count
    
    return total_correct / len(y_true)

def cluster_metrics(y_true, y_pred):
    """
    Compute all clustering metrics comprehensively.
    """
    results = {
        'ACC': acc(y_true, y_pred),
        'NMI': nmi(y_true, y_pred),
        'VME': vmeasure(y_true, y_pred),
        'ARI': ari(y_true, y_pred),
        'PUR': purity(y_true, y_pred)
    }
    return results

def test(y_true, y_pred):
    """
    Print all clustering metrics.
    """
    metrics = cluster_metrics(y_true, y_pred)
    print("ACC:%.4f, NMI:%.4f, VME:%.4f, ARI:%.4f, PUR:%.4f" % (
        metrics['ACC'], metrics['NMI'], metrics['VME'], metrics['ARI'], metrics['PUR']))
    return metrics['NMI']