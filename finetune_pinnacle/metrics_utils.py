from typing import Dict
import numpy as np
import pandas as pd

import json, matplotlib, os

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')

from data_prep import process_and_split_data


def save_results(output_results_path: str, ap_scores: Dict[str, Dict[str, float]], auroc_scores: Dict[str, Dict[str, float]]):
    """
    Save results in the form of dictionary to a json file.
    """
    res = {'ap':ap_scores, 'auroc':auroc_scores}
    with open(output_results_path, 'w') as f:
        json.dump(res, f)        
    print(f"\nResults output to -> {output_results_path}")


def save_plots(output_figs_path: str, positive_proportion_train: Dict[str, Dict[str, float]], positive_proportion_test: Dict[str, Dict[str, float]], ap_scores: Dict[str, Dict[str, float]], auroc_scores: Dict[str, Dict[str, float]], disease, wandb):
    """
    Render and save/log plots.
    """
    for eval_results, eval_type in zip([ap_scores, auroc_scores], ['ap_scores', 'auroc_scores']):
        i = 0
        fig = plt.figure(figsize=(10 * len(eval_results), 6 * len(eval_results)))
        for disease, ct_res in eval_results.items():
            all_scores = []
            xlabels = []
            i += 1
            for celltype, score in ct_res.items():  # No repetition of experiments, so it's score but not scores for each cell type
                all_scores = all_scores + [score]
                xlabels = xlabels + [celltype]
            ax = plt.subplot(len(eval_results), 1, i)
            plot_data = pd.DataFrame({'y': xlabels, 'x': all_scores})
            sns.barplot(x='x', y='y', data=plot_data, capsize=.2)
            if eval_type == 'ap_scores':
                try:
                    plt.scatter(y = xlabels, 
                                x = [list(positive_proportion_train[disease].values())[0]] * (len(positive_proportion_test[disease])-1) + [list(positive_proportion_train[disease].values())[1]], 
                                c = 'grey', zorder = 100, label = 'train', s = 10)  # np.arange(len(positive_proportion_test[disease]))
                except:
                    plt.scatter(y = xlabels, 
                                x = [list(positive_proportion_train[disease].values())[0]] * (len(positive_proportion_test[disease])), 
                                c = 'grey', zorder = 100, label='train', s = 10)  # np.arange(len(positive_proportion_test[disease]))
                plt.scatter(y = xlabels,
                            x = positive_proportion_test[disease].values(), 
                            c = 'black', zorder = 100, label = 'test', s = 10)
            ax.set_ylabel("")
            ax.set_xlabel(disease + '-' + eval_type)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        plt.tight_layout()
        plt.savefig(output_figs_path + eval_type + '.png')
        wandb.log({f'test {eval_type} bar':fig})
        plt.close(fig)
    return


def save_torch_train_val_preds(best_train_y, best_train_preds, best_train_groups, best_train_cts, best_val_y, best_val_preds, best_val_groups, best_val_cts, groups_map_train, groups_map_val, cts_map_train, cts_map_val, models_output_dir, embed_name, disease, mod, wandb):
    train_ranks = {}
    val_ranks = {}
    for ct in np.unique(best_val_cts):
        hits = np.where(best_val_cts==ct)[0]
        val_y_ct = best_val_y[hits]
        val_preds_ct = best_val_preds[hits]
        val_groups_ct = best_val_groups[hits]
        ct_name = cts_map_val[ct]

        if len(np.unique(val_y_ct)) < 2:
            auroc_score, ap_score, ct_recall_5, ct_precision_5, ct_ap_5, ct_recall_10, ct_precision_10, ct_ap_10, ct_recall_20, ct_precision_20, ct_ap_20, sorted_val_y_ct, sorted_val_preds_ct, sorted_val_groups_ct, positive_proportion_val = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, np.array([-1] * len(val_y_ct)), np.array([-1] * len(val_y_ct)), np.array([-1] * len(val_y_ct)), -1
        else:
            auroc_score, ap_score, ct_recall_5, ct_precision_5, ct_ap_5, ct_recall_10, ct_precision_10, ct_ap_10, ct_recall_20, ct_precision_20, ct_ap_20, sorted_val_y_ct, sorted_val_preds_ct, sorted_val_groups_ct, positive_proportion_val = get_metrics(val_y_ct, val_preds_ct, val_groups_ct, "training")
            if len(sorted_val_y_ct) > 0: sorted_val_y_ct = sorted_val_y_ct.squeeze(-1)
            if len(sorted_val_preds_ct) > 0: sorted_val_preds_ct = sorted_val_preds_ct.squeeze(-1)
        
        temp = pd.DataFrame({'y':sorted_val_y_ct, 'preds':sorted_val_preds_ct, 'name':[groups_map_val[prot_ind] for prot_ind in sorted_val_groups_ct]})
        temp['type'] = ['val'] * len(temp)
        val_ranks[ct_name] = temp
        temp.to_csv(f'{models_output_dir}/{embed_name}_{disease}_{mod}_val_preds_{ct_name}.csv', index=False)  # Save the validation predictions

        wandb.log({f'val AUPRC cell types {ct_name}': ap_score, 
                    f'val AUROC cell types {ct_name}': auroc_score,
                    f'val recall@5 cell types {ct_name}': ct_recall_5,
                    f'val precision@5 cell types {ct_name}': ct_precision_5,
                    f'val AP@5 cell types {ct_name}': ct_ap_5,
                    f'val recall@10 cell types {ct_name}': ct_recall_10, 
                    f'val precision@10 cell types {ct_name}': ct_precision_10,
                    f'val AP@10 cell types {ct_name}': ct_ap_10,
                    f'val recall@20 cell types {ct_name}': ct_recall_20,
                    f'val precision@20 cell types {ct_name}': ct_precision_20,
                    f'val AP@20 cell types {ct_name}': ct_ap_20,
                    f'val positive proportion {ct_name}': positive_proportion_val})
            
    for ct in np.unique(best_train_cts):  # We don't want to mess up train & val, so better separate
        hits = np.where(best_train_cts==ct)[0]
        train_y_ct = best_train_y[hits]
        train_preds_ct = best_train_preds[hits]
        train_groups_ct = best_train_groups[hits]
        # ct_recall_5, ct_precision_5, ct_ap_5, _ = precision_recall_at_k(train_y_ct, train_preds_ct, k=5)
        # ct_recall_10, ct_precision_10, ct_ap_10, _ = precision_recall_at_k(train_y_ct, train_preds_ct, k=10)
        #_, _, _, (sorted_train_y_ct, sorted_train_preds_ct, sorted_train_groups_ct) = precision_recall_at_k(train_y_ct, train_preds_ct, k=20, prots=train_groups_ct)
        _, _, _, (sorted_train_y_ct, sorted_train_preds_ct, sorted_train_groups_ct) = precision_recall_at_k(train_y_ct, train_preds_ct, k=10, prots=train_groups_ct)

        ct_name = cts_map_train[ct]
        temp = pd.DataFrame({'y': sorted_train_y_ct.squeeze(-1), 'preds':sorted_train_preds_ct.squeeze(-1), 'name':[groups_map_train[prot_ind] for prot_ind in sorted_train_groups_ct]})
        temp['type'] = ['train'] * len(temp)
        train_ranks[ct_name] = temp
        temp.to_csv(f'{models_output_dir}/{embed_name}_{disease}_{mod}_train_preds_{ct_name}.csv', index=False)  # Save the validation predictions
    
    return train_ranks, val_ranks


def precision_recall_at_k(y: np.ndarray, preds: np.ndarray, k: int = 10, prots: np.ndarray = None):
    """ Calculate recall@k, precision@k, and AP@k for binary classification.
    """
    assert preds.shape[0] == y.shape[0]
    assert k > 0
    if k > preds.shape[0]: return -1, -1, -1, ([], [], [])
    
    # Sort the scores and the labels by the scores
    sorted_indices = np.argsort(preds.flatten())[::-1]
    sorted_preds = preds[sorted_indices]
    sorted_y = y[sorted_indices]
    if prots is not None:
        sorted_prots = prots[sorted_indices]
    else: sorted_prots = None
    
    # Get the scores of the k highest predictions
    topk_preds = sorted_preds[:k]
    topk_y = sorted_y[:k]
    
    # Calculate the recall@k and precision@k
    recall_k = np.sum(topk_y) / np.sum(y)
    precision_k = np.sum(topk_y) / k
    
    # Calculate the AP@k
    # print(topk_y, topk_preds)
    ap_k = average_precision_score(topk_y, topk_preds)

    return recall_k, precision_k, ap_k, (sorted_y, sorted_preds, sorted_prots)


def get_metrics(y, y_pred, groups, celltype):
    if celltype in ["training"]:
        y = {celltype: y}
        groups = {celltype: groups}
            
    auroc_score = roc_auc_score(y[celltype], y_pred)  # s[disease][celltype]
    ap_score = average_precision_score(y[celltype], y_pred)
    recall_5, precision_5, ap_5, _ = precision_recall_at_k(y[celltype], y_pred, k=5)
    #recall_10, precision_10, ap_10, _ = precision_recall_at_k(y[celltype], y_pred, k=10)
    recall_10, precision_10, ap_10, (sorted_y, sorted_preds, sorted_groups) = precision_recall_at_k(y[celltype], y_pred, k=10, prots=np.array(groups[celltype]))
    #recall_20, precision_20, ap_20, (sorted_y, sorted_preds, sorted_groups) = precision_recall_at_k(y[celltype], y_pred, k=20, prots=np.array(groups[celltype]))
    recall_20, precision_20, ap_20 = -1, -1, -1

    # Calculate positive label proportions for each cell type, i.e. baseline for AP metric
    positive_proportion = sum(y[celltype]) / len(y[celltype])

    return auroc_score, ap_score, recall_5, precision_5, ap_5, recall_10, precision_10, ap_10, recall_20, precision_20, ap_20, sorted_y, sorted_preds, sorted_groups, positive_proportion