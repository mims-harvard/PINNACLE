from collections import Counter
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc

import networkx as nx
import obonet


def read_ts_data(f):
    ts_data = sc.read_h5ad(f)
    print(ts_data)
    print(ts_data.obs[["organ_tissue", "cell_ontology_class", "compartment"]])
    print(ts_data.obs["cell_ontology_class"].unique())

    return ts_data


def count_cells_per_celltype(f):
    ts_data_tissue = pd.read_csv(f, sep = "\t")
    print(ts_data_tissue)
    return Counter(ts_data_tissue["cell_ontology_class"].tolist())


def get_dendrogram(ts_data):
    print(ts_data.uns["dendrogram_cell_type_tissue"])
    print(ts_data.uns["dendrogram_tissue_cell_type"])


def load_global_PPI(f):
    G = nx.read_edgelist(f)
    print("Number of nodes:", len(G.nodes))
    print("Number of edges:", len(G.edges))
    return G


#def load_celltype_ppi(ppi_dir):
#    ppi_layers = dict()
#
#    for f in glob.glob(ppi_dir + "*"): # Expected format of filename: <PPI_DIR>/<CONTEXT>.<suffix>
#
#        # Parse name of context
#        context = f.split(ppi_dir)[1].split(".")[0]
#        context = context.replace("_", " ")
#
#        # Read edgelist
#        ppi = nx.read_edgelist(f)
#        ppi_layers[context] = ppi
#        assert nx.is_connected(ppi_layers[context])


def load_celltype_ppi(f):
    ppi_layers = dict()
    with open(f) as fin:
        for lin in fin:
            cluster = (lin.split("\t")[0], lin.split("\t")[1])
            ppi_layers[cluster] = lin.strip().split("\t")[2].split(",")
    print(ppi_layers.keys())
    return ppi_layers


def jaccard_similarity(set1, set2):
    return len(set1.intersection(set2)) / len(set1.union(set2))


def ontology_distance(tree, a, b, diameter):
    if not nx.has_path(tree, a, b): return None
    return nx.shortest_path_length(tree, a, b) / diameter


def read_obo(f):
    return obonet.read_obo(f)


def calculate_correlation(x, y, title, x_label, y_label, plot_f):
    x_ordered = []
    y_ordered = []
    for pair in x:
        x_ordered.append(x[pair])
        y_ordered.append(y[pair])

    ax = sns.regplot(x = x_ordered, y = y_ordered, marker = ".", y_jitter = 0.005, scatter_kws={"color": "silver"}, line_kws={"color": "darkcyan"})
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(plot_f)
    plt.close()
    return spearmanr(x_ordered, y_ordered)


def plot_box(data, dodge, x, y, title, x_label, y_label, legend, plot_f):
    ax = sns.boxplot(data = data, x = x, y = y, hue = x, dodge = dodge)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    if not legend: ax.get_legend().remove()
    plt.savefig(plot_f)
    plt.close()
