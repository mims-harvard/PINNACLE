import random
import glob
import argparse
import pandas as pd
import numpy as np

from collections import Counter

import scanpy as sc

import networkx as nx

from utils import load_global_PPI, read_ts_data, count_cells_per_celltype
from utils import calculate_correlation


TABULA_SAPIENS_DIR = "/n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/data/TabulaSapiens.h5ad"
PPI_DIR = "/n/data1/hms/dbmi/zitnik/lab/datasets/2020-12-PPI/processed/ppi_edgelist.txt"
OUTPUT_DIR = "/n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/processed/"


def subsample_cells(ts_data, groupby, num_cells_cutoff):
    num_valid_cells = 0
    new_data = []
    for cell, num in Counter(ts_data.obs[groupby].tolist()).items():
        print(cell, num)
        if num >= num_cells_cutoff:
            cell_data = ts_data[ts_data.obs[groupby] == cell]
            sample_idx = random.sample(range(num), num_cells_cutoff)
            sample = cell_data[sample_idx]
            if len(new_data) == 0: new_data = sample
            else: new_data = new_data.concatenate(sample, join = "inner")
            num_valid_cells += 1
    print("Keeping %d cell types" % num_valid_cells)
    return new_data


def rank_by_stattest(ts_data, ppi, method, groupby, out_f):
    print("Ranking using %s method by `%s`, which has %d unique classes..." % (method, groupby, len(ts_data.obs[groupby].unique())))
    sc.tl.rank_genes_groups(ts_data, groupby, method = method)
    
    print("Saving ranks h5ad in %s..." % (out_f + ".h5ad"))
    ts_data.write(out_f + ".h5ad")
    
    print("Saving ranks csv in %s..." % (out_f + ".csv"))
    ranked = ts_data.uns["rank_genes_groups"]
    groups = ranked["names"].dtype.names
    rank_pval = pd.DataFrame({group + "_" + key[:1]: ranked[key][group] for group in groups for key in ["names", "pvals"]})
    print(rank_pval)
    rank_pval.to_csv(out_f + ".csv", sep = "\t")

    return ts_data, rank_pval


def extract_celltype_ppi(input_f, output_f, ppi, lcc = True, max_pval = 1, max_number_of_genes = 4000):
    rank_pval = pd.read_csv(input_f, sep = "\t").drop("Unnamed: 0", axis = 1)

    celltype_ppi = dict.fromkeys([c.split("_")[0] for c in rank_pval.columns])
    
    if output_f != "":
        out = open(output_f + "=%s.csv" % str(max_pval), "w")

    for i, c in enumerate(sorted(celltype_ppi)):
        c_df = rank_pval[[c + "_n", c + "_p"]]
        c_df_sig = c_df[c_df[c + "_p"] <= max_pval]
        c_df_sig = c_df_sig.iloc[0:max_number_of_genes]
        if len(c_df_sig) == 0: continue

        c_df_sig_genes = c_df_sig[c + "_n"].tolist()
        c_subgraph = ppi.subgraph(c_df_sig_genes)
        if lcc:
            c_lcc = max(nx.connected_components(c_subgraph), key=len)
            print("Size of LCC for %s:" % c, len(c_lcc), "(%.2f of sig genes in PPI)" % (len(c_lcc) / len(c_subgraph)))
            if len(c_lcc) < 1000: continue

        else:
            c_lcc = c_subgraph.nodes()
        
        sorted_c_lcc = [n for n in c_df_sig_genes if n in c_lcc]
        celltype_ppi[c] = sorted_c_lcc
        nodes = ",".join(sorted_c_lcc)
        if output_f != "":
            out.write("\t".join([str(i), c, nodes]) + "\n")

    return celltype_ppi


def aggregate_celltype_ppi_list(celltypes, celltype_ppi_list, ppi, cells_per_celltype, out_f):
    print("Writing to...%s" % out_f)
    out = open(out_f, "w")
    
    frac_nodes = dict()
    kept_cells_per_celltype = dict()
    aggregated = dict()
    for i, c in enumerate(celltypes):        
        genes = []
        num_celltypes_found = 0
        for ppi_dict in celltype_ppi_list:
            if c in ppi_dict and ppi_dict[c] != None:
                genes.extend(ppi_dict[c])
                num_celltypes_found += 1
        if len(genes) == 0: continue
        gene_counts = Counter(genes)
        keep_genes = [g for g in gene_counts if gene_counts[g] / num_celltypes_found >= 0.9]
        print("Kept genes:", len(keep_genes), "out of", len(set(genes)))
        frac_nodes[c] = len(keep_genes) / len(set(genes))
        kept_cells_per_celltype[c] = cells_per_celltype[c]
        
        c_subgraph = ppi.subgraph(keep_genes)
        c_lcc = max(nx.connected_components(c_subgraph), key=len)
        print("Kept LCC:", len(c_lcc), "out of", len(set(keep_genes)))
        
        if len(c_lcc) > 1000:
            aggregated[c] = c_lcc
            nodes = ",".join(c_lcc)
            out.write("\t".join([str(i), c, nodes]) + "\n")
    
    calculate_correlation(kept_cells_per_celltype, frac_nodes, "Fraction of nodes kept across >= 0.9 iterations", "Number of cells in original cell type", "Fraction of nodes kept", "figures/frac_nodes_per_cells.pdf")

    for c, p in aggregated.items():
        print(c, len(p))
    print("Number of cell types:", len(aggregated))


def main():

    parser = argparse.ArgumentParser(description="Constructing cell type specific PPI networks.")
    parser.add_argument("-rank", type=bool, help="Perform cell type specific gene ranking.")
    parser.add_argument("-annotation", type=str, default="cell_ontology_class", help="Column for cell type annotation.")
    parser.add_argument("-subsample", type=bool, default=False, help="Subsample equal number of cells per cluster.")
    parser.add_argument("-num_cells_cutoff", type=int, default=100, help="Number of cells to subsample (and cutoff for cell types).")
    parser.add_argument("-iterations", type=int, default=10, help="Iterations to subsample.")
    parser.add_argument("-rank_pval_filename", type=str, default=OUTPUT_DIR + "ranked_TabulaSapiens", help="Filename (prefix) of ranked genes and their p-values.")
    parser.add_argument("-max_pval", type=float, default=1, help="Maximum p-value threshold for ranked genes.")
    parser.add_argument("-max_num_genes", type=int, default=4000, help="Maximum number of genes to keep (pre-LCC).")
    parser.add_argument("-celltype_ppi_filename", type=str, default=OUTPUT_DIR + "ppi_TabulaSapiens", help="Filename (prefix) of cell type specific PPI.")
    args = parser.parse_args()

    # Read global PPI
    print("Loading in global PPI network...")
    ppi = load_global_PPI(PPI_DIR)
    print("Finished loading in `ppi`...")

    # If ranking has not already been done...
    if args.rank:

        print("Loading in Tabula Sapiens data...")
        ts_data = read_ts_data(TABULA_SAPIENS_DIR)
        print("Finished loading `ts_data`...")

        if args.subsample:
            print("Ranking genes per cell type (subsampled)...")
            for i in range(args.iterations):
                i_data = subsample_cells(ts_data, args.annotation, args.num_cells_cutoff)
                if args.iterations > 1: out_f = args.rank_pval_filename + ("_%d" % i)
                else: out_f = args.rank_pval_filename
                i_ts_data, i_rank_pval = rank_by_stattest(i_data, ppi, "wilcoxon", args.annotation, out_f)
                print(i_rank_pval)
        else:
            print("Ranking genes per cell type...")
            ts_data, rank_pval = rank_by_stattest(ts_data, ppi, "wilcoxon", args.annotation, args.rank_pval_filename)
        print("Finished ranking genes...")
    
    # Extract cell type specific PPI
    else:
        
        cells_per_celltype = count_cells_per_celltype(OUTPUT_DIR + "ts_data_tissue.csv")
        print(cells_per_celltype)

        print("Extracting cell type specific PPI...")
        if args.subsample:
            
            celltypes = set()
            celltype_ppi_list = []
            for i, f in enumerate(glob.glob(args.rank_pval_filename + "*.csv")):
                print(f)
                i_ppi = extract_celltype_ppi(f, "", ppi, lcc = False, max_pval = args.max_pval, max_number_of_genes = args.max_num_genes)
                celltype_ppi_list.append(i_ppi)
                celltypes = celltypes.union(set(list(i_ppi.keys())))
            
            aggregate_celltype_ppi_list(celltypes, celltype_ppi_list, ppi, cells_per_celltype, args.celltype_ppi_filename + ("_maxpval=%s.csv" % str(args.max_pval)))
        else:
            extract_celltype_ppi(args.rank_pval_filename + ".csv", args.celltype_ppi_filename + "_maxpval", ppi, lcc = True, max_pval = args.max_pval, max_number_of_genes = args.max_num_genes)
    
    print("All finished!")
    

if __name__ == "__main__":
    main() 
