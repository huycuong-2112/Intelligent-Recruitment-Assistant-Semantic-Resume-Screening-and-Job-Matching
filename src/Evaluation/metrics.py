import numpy as np
from sklearn.metrics import ndcg_score

threshold = 2.0

def precision_at_k(y_true_rel, y_score, k, rel_threshold):
    if len(y_true_rel) == 0:
        return 0.0
    
    actual_k = min(k, len(y_true_rel))

    top_k_indices = np.argsort(y_score)[::-1][:k]
    
    relevant_in_top_k = np.sum(y_true_rel[top_k_indices] >= rel_threshold)
    
    return float(relevant_in_top_k / actual_k)

def recall_at_k(y_true_rel, y_score, k, rel_threshold):
    total_relevant = np.sum(y_true_rel >= rel_threshold)
    if total_relevant == 0:
        return 0.0
    
    actual_k = min(k, len(y_true_rel))
    
    top_k_indices = np.argsort(y_score)[::-1][:k]
    relevant_in_top_k = np.sum(y_true_rel[top_k_indices] >= rel_threshold)
    
    return float(relevant_in_top_k / total_relevant)

def ndcg_at_k(y_true_rel, y_score, k):
    if len(y_true_rel) == 0 or np.sum(y_true_rel) == 0:
        return 0.0
    
    actual_k = min(k, len(y_true_rel))

    y_true_2d = np.array([y_true_rel])
    y_score_2d = np.array([y_score])

    return float(ndcg_score(y_true_2d, y_score_2d, k=actual_k))