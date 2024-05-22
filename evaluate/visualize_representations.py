import glob
from collections import Counter
import os
import numpy as np
import pandas as pd
import json
import networkx as nx
import umap
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px

import torch

FIGSIZE = (15, 15)

def read_ppi(ppi_dir):
    ppi_layers = dict()
    for f in glob.glob(ppi_dir + "*"): # Expected format of filename: <PPI_DIR>/<CONTEXT>.<suffix>
        context = f.split(ppi_dir)[1].split(".")[0]
        ppi = nx.read_edgelist(f)
        ppi_layers[context] = ppi
        assert nx.is_connected(ppi_layers[context])
    return ppi_layers


def read_embed(ppi_embed_f, mg_embed_f, labels_f):

    # Read embeddings
    ppi_embed = torch.load(ppi_embed_f)
    embeddings = []
    for celltype, x in ppi_embed.items():
        embeddings.append(x)
    embeddings = torch.cat(embeddings)
    print("PPI embeddings", embeddings.shape)

    # Read metagraph embeddings
    mg_embed = torch.load(mg_embed_f)
    print("Meta graph embeddings", mg_embed.shape)

    # Read labels
    with open(labels_f) as f:
        labels_dict = f.read()
    labels_dict = labels_dict.replace("\'", "\"")
    labels_dict = json.loads(labels_dict)

    return embeddings, mg_embed, labels_dict


def plot_PINNACLE_embeddings(labels_dict, embedding, metagraph, color_map, n_neighbors, min_dist, do_plot_metagraph, output_dir):
    
    # Plot all embeddings
    protein_labels = dict()
    if do_plot_metagraph:
        protein_labels["x"] = embedding[len(metagraph.nodes):, 0]
        protein_labels["y"] = embedding[len(metagraph.nodes):, 1]
    else:
        protein_labels["x"] = embedding[:, 0]
        protein_labels["y"] = embedding[:, 1]
    celltype_name = []
    node_type = []
    node_name = []
    for n, p in zip(labels_dict["Cell Type"], labels_dict["Name"]):
        if not n.startswith("CCI_") and not n.startswith("BTO"):
            node_type.append("point")
            celltype_name.append(n)
            node_name.append(p)
    protein_labels["Cell Type"] = celltype_name
    protein_labels["Node Type"] = node_type
    protein_labels["Node Name"] = node_name
    plot_protein_umap(protein_labels, color_map, output_dir + "umap.all=%s_mindist=%s" % (n_neighbors, min_dist))
    
    # Plot metagraph embeddings only
    if do_plot_metagraph:
        mg_labels = dict()
        mg_labels["x"] = embedding[0:len(metagraph.nodes), 0]
        mg_labels["y"] = embedding[0:len(metagraph.nodes), 1]
        celltype_name = []
        node_type = []
        for n in (labels_dict["Name"][0:len(metagraph.nodes)]):
            if n.startswith("CCI_"):
                celltype_name.append(n.split("CCI_")[1])
                node_type.append("circle_down")
            else:
                node_type.append("circle_up")
                celltype_name.append(n)
        mg_labels["Cell Type"] = celltype_name
        mg_labels["Node Type"] = node_type
        plot_metagraph_umap(mg_labels, protein_labels, color_map, output_dir + "umap.ccibto=%s_mindist=%s" % (n_neighbors, min_dist))


def plot_emb(ppi_x, mg_x, labels_dict, ppi_layers, metagraph, umap_param, do_plot_metagraph, do_sweep, do_sweep_plot, output_dir):
    color_map = {n: c for n, c in zip(metagraph.nodes, sns.color_palette("husl", len(metagraph.nodes)))} # TODO
    if do_plot_metagraph:
        embed = torch.cat([ppi_x, mg_x])
    else:
        embed = ppi_x
    
    # Fit (and plot) a range of UMAP parameters
    if do_sweep:
        for n_neighbors in umap_param["n_neighbors"]:
            for min_dist in umap_param["min_dist"]:
                outfile = output_dir + ("embedding_nneighbors=%s_mindist=%s.npy" % (n_neighbors, min_dist))
                if not os.path.exists(outfile):
                    mapping, embedding = fit_umap(embed, n_neighbors, min_dist)
                    np.save(outfile, embedding)
                else:
                    embedding = np.load(outfile)
                    print("UMAP:", embedding.shape)

                if do_sweep_plot:
                    plot_PINNACLE_embeddings(labels_dict, embedding, metagraph, color_map, n_neighbors, min_dist, do_plot_metagraph, output_dir)

    # Fit and plot desired UMAP parameters
    else:
        outfile = output_dir + ("embedding_nneighbors=%s_mindist=%s.npy" % (umap_param["n_neighbors"], umap_param["min_dist"]))
        if not os.path.exists(outfile):
            print("Fit UMAP...")
            mapping, embedding = fit_umap(embed, umap_param["n_neighbors"], umap_param["min_dist"])
            np.save(outfile, embedding)
        else:
            embedding = np.load(outfile)
            print("UMAP:", embedding.shape)
        
        plot_PINNACLE_embeddings(labels_dict, embedding, metagraph, color_map, umap_param["n_neighbors"], umap_param["min_dist"], do_plot_metagraph, output_dir)


def fit_umap(embed, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=3):
    mapping = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric, random_state=random_state).fit(embed)
    embedding = mapping.transform(embed)
    print("UMAP reduced:", embedding.shape)
    return mapping, embedding


def plot_protein_umap(labels, color_map, output):

    for k, v in labels.items():
        print(k, len(v))

    hover_keys = [l for l in labels.keys() if l != "Color"]
    df = pd.DataFrame(labels)
    
    plt.figure(figsize = FIGSIZE)
    ax = sns.scatterplot(data = df, x = "x", y = "y", hue = "Cell Type", palette = color_map, marker = ".", size = 1, linewidth = 0, alpha = 0.4, legend = False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output + ".pdf")
    plt.close()

    # Plot individual cell type contexts
    for celltype in ["medullary thymic epithelial cell", "bronchial vessel endothelial cell", "lung microvascular endothelial cell", "retinal blood vessel endothelial cell", "kidney epithelial cell", "tongue muscle cell", "cell of skeletal muscle", "mesenchymal stem cell", "fibroblast of breast", "fibroblast of cardiac tissue", "b cell", "cd4-positive helper t cell"]:
        print(celltype)
        celltype_colormap = {k: "#a3a2a2" if k != celltype else v for k, v in color_map.items()}
        fig, ax = plt.subplots(figsize = FIGSIZE)
        sns.scatterplot(data = df[df["Cell Type"] != celltype], x = "x", y = "y", hue = "Cell Type", marker = ".", palette = celltype_colormap, linewidth = 0, alpha = 0.2, ax = ax, legend = False)
        sns.scatterplot(data = df[df["Cell Type"] == celltype], x = "x", y = "y", hue = "Cell Type", marker = "o", palette = celltype_colormap, linewidth = 0, alpha = 1, ax = ax, legend = False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        plt.xlabel("")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(output + "_" + "_".join(celltype.split(" ")) + ".pdf")
        plt.close()
    
    fig = px.scatter(df, x = "x", y = "y", color = "Node Type", color_discrete_map = color_map, hover_data = hover_keys)
    fig.write_html(output + ".html")
    plt.close()


def plot_metagraph_umap(labels, protein_labels, color_map, output):

    for k, v in labels.items():
        print(k, len(v))

    hover_keys = [l for l in labels.keys() if l != "Color"]
    df = pd.DataFrame(labels)
    
    fig, ax = plt.subplots(figsize = FIGSIZE)
    sns.scatterplot(data = pd.DataFrame(protein_labels), x = "x", y = "y", marker = ".", color = "#a3a2a2", linewidth = 0, alpha = 0.2, legend = False)
    sns.scatterplot(data = df[~df["Cell Type"].str.contains("BTO")], x = "x", y = "y", marker = "^", color = "seagreen", linewidth = 0, alpha = 0.8, legend = False)
    sns.scatterplot(data = df[df["Cell Type"].str.contains("BTO")], x = "x", y = "y", marker = "s", color = "steelblue", linewidth = 0, alpha = 0.8, legend = False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output + ".pdf")
    plt.close()

    fig = px.scatter(df, x = "x", y = "y", color = "Cell Type", color_discrete_map = color_map, hover_data = hover_keys)
    fig.write_html(output + ".html")
    plt.close()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--do_sweep', type=bool, default=False)
    parser.add_argument('--do_sweep_plot', type=bool, default=False)
    parser.add_argument('--do_plot_metagraph', type=bool, default=False)
    args = parser.parse_args()

    input_f = "../data/pinnacle_embeds/"
    output_dir = "figures/"

    print("Read in data...")
    ppi_x, mg_x, labels_dict = read_embed(input_f + "pinnacle_protein_embed.pth", input_f + "pinnacle_mg_embed.pth", input_f + "pinnacle_labels_dict.txt")
    ppi_layers = read_ppi("../data/networks/ppi_edgelists/")
    metagraph = nx.read_edgelist("../data/networks/mg_edgelist.txt", delimiter = "\t")
    
    # Remove sanity check stuff
    sanity_idx = [i for i, l in enumerate(labels_dict["Cell Type"]) if "Sanity" not in l]
    new_labels_dict = dict()
    for k, v in labels_dict.items():
        new_labels_dict[k] = np.array(v)[sanity_idx]

    print("Plot embeddings...")
    if args.do_sweep:
        umap_param = {"n_neighbors": [10, 20, 30, 40, 50, 100], "min_dist": [0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9]}
    else:
        umap_param = {"n_neighbors": 10, "min_dist": 0.9}
    plot_emb(ppi_x, mg_x, new_labels_dict, ppi_layers, metagraph, umap_param, args.do_plot_metagraph, args.do_sweep, args.do_sweep_plot, output_dir)


if __name__ == "__main__":
    main()

