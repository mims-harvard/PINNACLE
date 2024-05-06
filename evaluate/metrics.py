import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, average_precision_score



def precision_recall_at_k(y: np.ndarray, preds: np.ndarray, k: int = 10):
    """
    Calculate recall@k and precision@k for binary classification.
    """
    assert preds.shape[0] == y.shape[0]
    assert k > 0
    if k > preds.shape[0]: return -1, -1, -1, -1
    
    # Sort the scores and the labels by the scores
    sorted_indices = np.argsort(preds.flatten())[::-1]
    sorted_preds = preds[sorted_indices]
    sorted_y = y[sorted_indices]
    
    # Get the scores of the k highest predictions
    topk_preds = sorted_preds[:k]
    topk_y = sorted_y[:k]
    
    # Calculate the recall@k and precision@k
    recall_k = np.sum(topk_y) / np.sum(y)
    precision_k = np.sum(topk_y) / k
    
    # Calculate the accuracy@k
    accuracy_k = accuracy_score(topk_y, topk_preds > 0.5)

    # Calculate the AP@k
    ap_k = average_precision_score(topk_y, topk_preds)

    return recall_k, precision_k, accuracy_k, ap_k
