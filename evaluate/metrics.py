import pandas as pd
import numpy as np

from scipy.stats import percentileofscore
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


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


def calculate_celltype_percentiles(model_outputs_df):
    df_percentiles = []
    for c in model_outputs_df["celltype"].unique():
        df = model_outputs_df[model_outputs_df["celltype"] == c]
        df["percentile"] = percentileofscore(df["preds"].tolist(), df["preds"].tolist())
        df_percentiles.append(df)
    df_percentiles = pd.concat(df_percentiles)
    return df_percentiles


def calculate_metrics(k, column, test_proteins, model_outputs_df):
    ap = dict()
    roc = dict()
    recall_k = dict()
    precision_k = dict()
    accuracy_k = dict()
    ap_k = dict()
    for col_item in model_outputs_df[column].unique(): # Iterate through cell types or benchmark models
        df = model_outputs_df[model_outputs_df[column] == col_item]
        
        # Calculate overall AUROC and AP for the cell type
        auroc_score = roc_auc_score(df["y"].tolist(), df["preds"].tolist())
        ap_score = average_precision_score(df["y"].tolist(), df["preds"].tolist())
        
        # Calculate metrics at k
        recall, precision, accuracy, ap_k_score = precision_recall_at_k(np.array(df["y"].tolist()), np.array(df["preds"].tolist()), k = k)

        # Save metrics
        ap[col_item] = ap_score
        roc[col_item] = auroc_score
        recall_k[col_item] = recall
        precision_k[col_item] = precision
        accuracy_k[col_item] = accuracy
        ap_k[col_item] = ap_k_score

    return ap, roc, recall_k, precision_k, accuracy_k, ap_k
