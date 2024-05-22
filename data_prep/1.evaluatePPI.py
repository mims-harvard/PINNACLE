# Run after 0.constructPPI.py

import argparse
import glob
import pandas as pd
import numpy as np
from operator import itemgetter

import networkx as nx

from utils import read_ts_data, load_global_PPI, read_obo, load_celltype_ppi
from utils import jaccard_similarity, ontology_distance, calculate_correlation, plot_box

import sys
sys.path.insert(0, '..') # add data_config to path
from data_config import PPI_DIR, CELL_ONTOLOGY_DIR


def evaluate(ppi_layers, ppi, cell_ontology, cell_ontology_names):
    print("Number of cell types:", len(ppi_layers))

    lcc = []
    freq_genes = dict.fromkeys(ppi.nodes, set())
    top_centrality = dict.fromkeys(ppi_layers.keys(), set())
    all_genes = set()
    for c, p in ppi_layers.items():
        c = c[1]
        lcc.append(len(p))
        all_genes = all_genes.union(set(p))

        # How many cell types does a gene appear in?
        for g in p:
            freq_genes[g].add(c)

        # Overlap of top 100 network centrality genes for immune vs. non-immune compartments?
        p_pagerank = nx.pagerank(ppi.subgraph(p))
        top_centrality[c] = dict(sorted(p_pagerank.items(), key = itemgetter(1), reverse = True)[:100])

    print("Minimum LCC:", min(lcc), "Maximum LCC:", max(lcc), "Average LCC:", np.mean(lcc), "+/-", np.std(lcc))
    calculate_genome_coverage(all_genes, ppi)
    #calculate_gene_overlap(freq_genes)
    calculate_celltype_jaccard(ppi_layers, top_centrality, cell_ontology, cell_ontology_names)


def calculate_genome_coverage(all_genes, ppi):
    print("Unique genes in cell type PPI networks:", len(all_genes))
    print("Unique genes in global PPI network:", len(ppi))
    print("Coverage", len(all_genes) / len(ppi))


def shortest_path_to_root(celltypes, cell_ontology, cell_ontology_names):
    root = "CL:0000000"
    spls = []
    for c_1 in celltypes:
        c_1 = c_1[1]
        for c_2 in celltypes:
            c_2 = c_2[1]
            if c_1 in cell_ontology_names and c_2 in cell_ontology_names:
                c1_to_root = nx.shortest_path_length(cell_ontology, cell_ontology_names[c_1], root)
                c2_to_root = nx.shortest_path_length(cell_ontology, cell_ontology_names[c_2], root)
                spls.append(c1_to_root + c2_to_root)
    print("Minimum SPLs:", min(spls), "Maximum SPLs:", max(spls), "Average SPLs:", np.mean(spls), "+/-", np.std(spls))


def calculate_gene_overlap(freq_genes):
    overlap = []
    for g, celltypes in freq_genes.items():
        #print(g, len(celltypes))
        overlap.append(len(celltypes))
    print("On average, genes appear in %.2f cell types" % np.mean(overlap))


def calculate_celltype_jaccard(ppi_layers, top_centrality, tree, tree_names):
    tree = tree.to_undirected()
    print("Tree is now undirected")
    
    jaccard_sim = dict()
    jaccard_list = []
    top100_jaccard_sim = dict()
    top100_jaccard_list = []
    top100_semantic_list = []
    semantic_sim = dict()
    semantic_list = []
    diameter = 1 #nx.diameter(tree) #max([nx.diameter(tree.subgraph(cc)) for cc in nx.strongly_connected_components(tree)])
    for c_1, ppi_1 in ppi_layers.items():
        c_1 = c_1[1]
        top100_ppi_1 = top_centrality[c_1]
        for c_2, ppi_2 in ppi_layers.items():
            c_2 = c_2[1]
            top100_ppi_2 = top_centrality[c_2]
            if c_1 in tree_names and c_2 in tree_names:
                if not nx.has_path(tree, tree_names[c_1], tree_names[c_2]):
                    print("No path found between %s and %s" % (c_1, c_2))
                    continue
                semantic_sim[(c_1, c_2)] = ontology_distance(tree, tree_names[c_1], tree_names[c_2], diameter)
                jaccard_sim[(c_1, c_2)] = jaccard_similarity(set(ppi_1), set(ppi_2))
                semantic_list.append(semantic_sim[(c_1, c_2)])
                jaccard_list.append(jaccard_sim[(c_1, c_2)])
                if len(top100_ppi_1) == 0 and len(top100_ppi_2) == 0: continue
                top100_jaccard_sim[(c_1, c_2)] = jaccard_similarity(set(top100_ppi_1), set(top100_ppi_2))
                top100_jaccard_list.append(top100_jaccard_sim[(c_1, c_2)])
                top100_semantic_list.append(semantic_sim[(c_1, c_2)])

    dodge = False
    legend = False
    semantic_jaccard_sims = pd.DataFrame.from_dict({"Semantic Distance": semantic_list, "Jaccard Similarity": jaccard_list})
    plot_box(semantic_jaccard_sims, dodge, "Semantic Distance", "Jaccard Similarity", "Jaccard Similarity vs. Semantic Distance", "Semantic Distance", "Jaccard Similarity", legend, "figures/ppi_semantic_jaccard_sim.pdf")
    semantic_top100jaccard_sims = pd.DataFrame.from_dict({"Semantic Distance": top100_semantic_list, "Jaccard Similarity": top100_jaccard_list})
    plot_box(semantic_top100jaccard_sims, dodge, "Semantic Distance", "Jaccard Similarity", "Top 100 Jaccard Similarity vs. Semantic Distance", "Semantic Distance", "Jaccard Similarity", legend, "figures/ppi_top100_semantic_jaccard_sim.pdf")


def main():

    parser = argparse.ArgumentParser(description="Evaluating cell type specific PPI networks.")
    parser.add_argument("-celltype_ppi", type=str, help="Filename (prefix) of cell type PPI.")
    args = parser.parse_args()

    # Read global PPI
    print("Loading in global PPI network...")
    ppi = load_global_PPI(PPI_DIR)
    print("Finished loading in `ppi`...")

    # Read cell type specific PPI
    print("Loading in cell type specific PPI...")
    celltype_ppi = load_celltype_ppi(args.celltype_ppi)
    print("Finished loading in cell type specific PPI...")

    # Read Cell Ontology
    print("Loading in Cell Ontology...")
    cell_ontology = read_obo(CELL_ONTOLOGY_DIR)
    cell_ontology_names = {str(data.get('name')).lower(): id_ for id_, data in cell_ontology.nodes(data=True)}
    ts_cell_ontology = []
    for c in celltype_ppi:
        if c[1] in cell_ontology_names:
            ts_cell_ontology.append(cell_ontology_names[c[1]])
        else:
            print(c)
    ts_cell_ontology = cell_ontology.subgraph(ts_cell_ontology)
    print(ts_cell_ontology.nodes())
    print("Number of nodes in Cell Ontology:", len(ts_cell_ontology.nodes))
    cell_ontology = cell_ontology.subgraph(max(nx.weakly_connected_components(cell_ontology), key=len))
    print("Number of nodes:", len(cell_ontology.nodes))
    print("Number of edges:", len(cell_ontology.edges))
    print("Finished loading in Cell Ontology...")

    # Evaluate cell type specific PPI
    print("Evaluating cell type specific PPI...")
    shortest_path_to_root(list(celltype_ppi.keys()), cell_ontology, cell_ontology_names)
    evaluate(celltype_ppi, ppi, cell_ontology, cell_ontology_names)
    print("All finished!")


if __name__ == "__main__":
    main()