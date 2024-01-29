import random
import numpy as np
import pandas as pd
from collections import Counter
import umap
import umap.plot
from matplotlib import pyplot as plt
import plotly.express as px

import torch
import torch_sparse
import torch.nn.functional as F
from torch.nn import Sigmoid
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, roc_curve, precision_recall_curve, silhouette_score, calinski_harabasz_score, davies_bouldin_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calc_individual_metrics(pred, y):
    try: 
        roc_score = roc_auc_score(y, pred)
    except ValueError: 
        roc_score = 0.5 
    ap_score = average_precision_score(y, pred)
    acc = accuracy_score(y, pred > 0.5)
    f1 = f1_score(y, pred > 0.5, average = 'micro')
    return roc_score, ap_score, acc, f1


def calc_metrics(mg_pred, mg_data, ppi_preds, ppi_data):
    all_roc = []
    all_ap = []
    all_acc = []
    all_f1 = []

    # Compute metrics for metagraph
    if len(mg_pred) > 0:
        roc_score, ap_score, acc, f1 = calc_individual_metrics(mg_pred.cpu().detach().numpy(), mg_data["y"].cpu().detach().numpy())
        all_roc.append(roc_score)
        all_ap.append(ap_score)
        all_acc.append(acc)
        all_f1.append(f1)

    # Compute metrics for PPI layers
    for celltype, ppi in ppi_preds.items():
        curr_roc_score, curr_ap_score, curr_acc, curr_f1 = calc_individual_metrics(ppi.numpy(), ppi_data[celltype]["y"].numpy())
        all_roc.append(curr_roc_score)
        all_ap.append(curr_ap_score)
        all_acc.append(curr_acc)
        all_f1.append(curr_f1)
    return np.average(all_roc), np.average(all_ap), np.average(all_acc), np.average(all_f1)


def metrics_per_rel(mg_pred, mg_data, ppi_preds, ppi_data, edge_attr_dict, celltype_map, log_f, wandb, split):
    
    celltype_map = {v: k for k, v in celltype_map.items()}

    # Compute metrics per rel for metagraph
    if len(mg_pred) > 0:
        for attr, idx in edge_attr_dict.items():
            mask = (mg_data["total_edge_type"].cpu().detach().numpy() == idx)
            if mask.sum() == 0: continue
            pred_per_rel = mg_pred.cpu().detach().numpy()[mask]
            y_per_rel = mg_data["y"].cpu().detach().numpy()[mask]
            roc_per_rel, ap_per_rel, acc_per_rel, f1_per_rel = calc_individual_metrics(pred_per_rel, y_per_rel)
            log_f.write("ROC for edge type {}: {:.5f}\n".format(attr, roc_per_rel))
            log_f.write("AP for edge type {}: {:.5f}\n".format(attr, ap_per_rel))
            log_f.write("ACC for edge type {}: {:.5f}\n".format(attr, acc_per_rel))
            log_f.write("F1 for edge type {}: {:.5f}\n".format(attr, f1_per_rel))
            wandb.log({"%s_%s_roc" % (attr, split): roc_per_rel, "%s_%s_ap" % (attr, split): ap_per_rel, "%s_%s_acc" % (attr, split): acc_per_rel, "%s_%s_f1" % (attr, split): f1_per_rel})

    # Compute metrics per rel per PPI layer
    for celltype, ppi in ppi_preds.items():
        for attr, idx in edge_attr_dict.items():
            mask = (ppi_data[celltype]["total_edge_type"].numpy() == idx)
            if mask.sum() == 0: continue
            pred_per_rel = ppi.cpu().detach().numpy()[mask]
            y_per_rel = ppi_data[celltype]["y"].numpy()[mask]
            roc_per_rel, ap_per_rel, acc_per_rel, f1_per_rel = calc_individual_metrics(pred_per_rel, y_per_rel)
            log_f.write("ROC for edge type {} in celltype {}: {:.5f}\n".format(attr, celltype_map[celltype], roc_per_rel))
            log_f.write("AP for edge type {} in celltype {}: {:.5f}\n".format(attr, celltype_map[celltype], ap_per_rel))
            log_f.write("ACC for edge type {} in celltype {}: {:.5f}\n".format(attr, celltype_map[celltype], acc_per_rel))
            log_f.write("F1 for edge type {} in celltype {}: {:.5f}\n".format(attr, celltype_map[celltype], f1_per_rel))
            wandb.log({"%s_%s_%s_roc" % (celltype_map[celltype], attr, split): roc_per_rel, "%s_%s_%s_ap" % (celltype_map[celltype], attr, split): ap_per_rel, "%s_%s_%s_acc" % (celltype_map[celltype], attr, split): acc_per_rel, "%s_%s_%s_f1" % (celltype_map[celltype], attr, split): f1_per_rel})


def construct_metapath(metapaths, edge_index, edge_type, num_nodes):
    unique_edge_types = edge_type.unique()

    adjs = {}
    for et in unique_edge_types:
        row, col = edge_index[:, edge_type == et]
        adj = torch_sparse.SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
        adjs[int(et)] = adj

    mp_adjs = []
    for metapath in metapaths:
        mp_adj = None
        for idx in metapath:
            if idx not in adjs: continue
            if mp_adj:
                mp_adj @= adjs[idx]
            else:
                mp_adj = adjs[idx]
        if mp_adj: mp_adjs.append(mp_adj)

    mp_adjs = [torch.stack([mp_adj.storage.row(), mp_adj.storage.col()]) for mp_adj in mp_adjs]
    return mp_adjs


@torch.no_grad()
def get_embeddings(model, ppi_x, mg_x, ppi_metapaths, mg_metapaths, ppi_edge_index, mg_edge_index, tissue_neighbors):
    model.eval()
    ppi_x, mg_x = model(ppi_x, mg_x, ppi_metapaths, mg_metapaths, ppi_edge_index, mg_edge_index, tissue_neighbors)
    return ppi_x, mg_x 


def plot_emb(best_ppi_x, best_mg_x, celltype_map, ppi_layers, metagraph, wandb, finetune_labels, plot=False):
    celltype_map = {v: k for k, v in celltype_map.items()}
    embed, labels_df, mg_labels = combine_embed(best_ppi_x, best_mg_x, celltype_map, ppi_layers, metagraph, finetune_labels)

    if plot:
        mapping, embedding = fit_umap(embed, min_dist=0.5)
        labels_df["x"] = embedding[:, 0]
        labels_df["y"] = embedding[:, 1]
        plot_umap(labels_df, wandb, "umap.all")
        if len(best_mg_x) > 0:
            mg_labels["x"] = embedding[0:len(celltype_map), 0]
            mg_labels["y"] = embedding[0:len(celltype_map), 1]
            plot_umap(mg_labels, wandb, "umap.ccibto")
        labels_df.pop("x")
        labels_df.pop("y")
    return labels_df


def combine_embed(ppi_embed, mg_embed, key, ppi_layers, metagraph, finetune_labels):
    labels_df = dict()
    mg_labels = dict()

    if len(mg_embed) > 0:

        # Set metagraph labels
        labels_df["Cell Type"] = ["CCI_" + v if "BTO" not in v else v for k, v in key.items()]
        mg_labels["Cell Type"] = [v if "BTO" not in v else v for k, v in key.items()]
        labels_df["Name"] = ["CCI_" + v if "BTO" not in v else v for k, v in key.items()]

        labels_df["Degree"] = [100] * len(metagraph.nodes) # Artificially increase size
        labels_df["Relative Degree"] = [1] * len(metagraph.nodes)
        mg_labels["Degree"] = [metagraph.degree[n] for n in metagraph.nodes]
        
        pcount_labels = [0] * len(key)
        combined = [mg_embed]

    else: 
        labels_df["Cell Type"] = []
        labels_df["Name"] = []
        labels_df["Degree"] = []
        labels_df["Relative Degree"] = []
        pcount_labels = []
        combined = []

    # Get per protein counts & node degrees
    protein_counts = []
    for cluster, ppi in ppi_layers.items():
        if cluster in key.values(): protein_counts += list(ppi)
    protein_counts = Counter(protein_counts)
    
    # Get labels for PPI
    sanity = dict()
    for celltype, x in ppi_embed.items():
        labels_df["Cell Type"] += [key[celltype]] * x.size(0)
        degrees = [ppi_layers[key[celltype]].degree[n] for n in ppi_layers[key[celltype]].nodes]
        labels_df["Degree"] += degrees
        labels_df["Relative Degree"] += [round(d / max(degrees), 5) for d in degrees]
        labels_df["Name"] += list(ppi_layers[key[celltype]].nodes)
        max_rank = len(ppi_layers[key[celltype]])
        combined.append(x)
        for protein in ppi_layers[key[celltype]]:
            pcount_labels.append(protein_counts[protein])
        sanity[key[celltype]] = torch.mean(x, 0)

    labels_df["Overlap"] = pcount_labels

    # Concatenate
    combined = torch.cat(combined)

    # Sanity check
    combined = torch.cat((combined, torch.stack(list(sanity.values()))))
    labels_df["Cell Type"] += ["Sanity Check %s" % k for k in sanity]
    labels_df["Degree"] += [100] * len(sanity)
    labels_df["Relative Degree"] += [1] * len(sanity)
    labels_df["Name"] += ["Sanity Check %s" % k for k in sanity]
    labels_df["Overlap"] += [0] * len(sanity)
    
    return combined, labels_df, mg_labels


def fit_umap(embed, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=3):
    mapping = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric, random_state=random_state).fit(embed)
    embedding = mapping.transform(embed)
    print("UMAP reduced:", embedding.shape)
    return mapping, embedding


def plot_umap(labels, wandb, wb_title, color_category="default", finetune_labels=[]):
    hover_keys = list(labels.keys())
    df = pd.DataFrame(labels)
    fig = px.scatter(df, x="x", y="y", color="Cell Type", size="Degree", hover_data=hover_keys)
    wandb.log({wb_title: fig})
    plt.close()

    
def calc_cluster_metrics(ppi_x: dict) -> tuple:
    """
    Calculate Calinski-Harabasz score and Davies-Bouldin score of PPI embeddings.
    
    :param ppi_x: PPI node embeddings output from the model.
    
    :return: calinski_harabasz, and davies_bouldin scores.
    """
    X = torch.cat(list(ppi_x.values())).detach().cpu().numpy()
    labels = np.concatenate([[key] * x.shape[0] for key, x in ppi_x.items()])
    
    if len(np.unique(labels))==1:
        return 0, 0

    return calinski_harabasz_score(X, labels), davies_bouldin_score(X, labels)