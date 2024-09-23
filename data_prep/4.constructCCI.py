import argparse
import glob
from collections import defaultdict

import pandas as pd
import networkx as nx


def parse_cpdb_output(f, cluster_adj, pvalue, cutoff):
    df = pd.read_csv(f, sep="\t")
    print(df)
    for col in df.iteritems():
        if "|" in col[0]:
            source = col[0].split("|")[0]
            target = col[0].split("|")[1]
            LRs = col[1][col[1] < pvalue].dropna()
            num_LR = len(LRs)
            #print(source, target, num_LR)

            if source in cluster_adj and num_LR >= cutoff: 
                if target in cluster_adj[source]: 
                    cluster_adj[source][target] += 1
                else: 
                    cluster_adj[source][target] = 1
            else:
                cluster_adj[source] = {target: 1}
    return cluster_adj


def count_majority(data, num_iters, threshold=0.9):
    edges = []
    for k, v in data.items():
        for i, i_value in v.items():
            if i_value / num_iters >= threshold:
                edges.append((k, i))

    G = nx.Graph()
    G.add_edges_from(edges)
    print(nx.info(G))
    return G


def generate_cci(files, pvalue=0.001, cutoff=1):
    cluster_adj = dict()
    for f in files:
        cluster_adj = parse_cpdb_output(f, cluster_adj, pvalue, cutoff)
    return cluster_adj


def main():

    parser = argparse.ArgumentParser(description="Extracting cell-cell interactions.")
    parser.add_argument("-cpdb_output", type=str, help="Directory of output files from CellPhoneDB.")
    parser.add_argument("-cci_edgelist", type=str, help="Filename of cell-cell interaction network.")
    parser.add_argument("-threshold", type=float, default=0.9, help="Minimum number of iterations for a cell-cell interaction to be significant.")
    parser.add_argument("-pval", type=float, default=0.001, help="P-value for a cell-cell interaction to be significant.")
    parser.add_argument("-cutoff", type=int, default=1, help="Minimum number of significant LR interactions for a pair of cell types to have an edge.")
    args = parser.parse_args()

    cpdb_files = glob.glob(args.cpdb_output + "*/pvalues.txt")
    print("Number of output files to parse:", len(cpdb_files))
    cci = generate_cci(cpdb_files, args.pval, args.cutoff)
    G = count_majority(cci, len(cpdb_files), args.threshold)

    nx.write_edgelist(G, args.cci_edgelist, data = False, delimiter = "\t")


if __name__ == "__main__":
    main() 
