import argparse
import glob
import json
import obonet
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px

import utils
from metrics import calculate_metrics, calculate_celltype_percentiles

import sys
sys.path.insert(0, '..') # add data_config to path
from data_config import DATA_DIR, TS_TISSUE_DATA_DIR, METAGRAPH_DIR


def read_model_data(model_outputs_dir, disease, test_only):
    
    # Read model outputs
    # Table columns: y,preds,name,type
    model_outputs_dict = dict()
    for f in glob.glob(model_outputs_dir + ("TS_%s" % disease) + "_torch_mlp_all_preds_*"):
        celltype = f.split("all_preds_")[1].split(".csv")[0]
        celltype_df = pd.read_csv(f)
        celltype_df["celltype"] = celltype
        celltype_df = utils.filter_model_data(celltype_df, test_only)
        model_outputs_dict[celltype] = celltype_df
    model_outputs_df = pd.concat(list(model_outputs_dict.values()))
    utils.check_no_leakage_protein_split(model_outputs_df) # Double check that there is no data leakage
    
    # Create dictionary of test proteins and their labels
    # Dictionary: key = protein name, value = label
    test_proteins = dict()
    if test_only:
        for k, v in zip(model_outputs_df["name"].tolist(), model_outputs_df["y"].tolist()):
            if k in test_proteins: assert test_proteins[k] == v
            else: test_proteins[k] = v

    return model_outputs_df, test_proteins


def read_benchmarks(inventory_f, disease, test_only, seed):
    if inventory_f == "": return []

    benchmarks_list = []
    with open(inventory_f, "r") as fin:
        for line in fin:
            fname = line.strip("\n")
            if not "/TS_seed=%d/" % seed in fname: continue
            if disease not in fname: continue
            print(fname)
            benchmark_name = fname.split("all_preds_")[1].split(".csv")[0]

            df = pd.read_csv(fname)
            df["benchmark"] = benchmark_name
            print(df)
            df = utils.filter_model_data(df, test_only)
            benchmarks_list.append(df)
    
    print("Number of benchmarks:", len(benchmarks_list))
    benchmarks_df = pd.concat(benchmarks_list)
    utils.check_no_leakage_protein_split(benchmarks_df)
    return benchmarks_df


def plot_compartment_performance_across_seeds(metric_by_compartment_seeds, metric_name, benchmarks, out_f):
    
    # Aggregate (mean) metrics
    metric = metric_by_compartment_seeds.groupby(by = ["celltype", "compartment"]).mean().reset_index()
    benchmark_metric = benchmarks.groupby(by = ["benchmark"]).mean().reset_index()
    
    # Plot PINNACLE results
    color_map = {"Epithelial": "steelblue", "Stromal": "gold", "Endothelial": "darkorchid", "Immune": "firebrick", "Immune-Stromal": "darkorange", "Stromal-Epithelial": "yellowgreen", "Germ line": "violet"}
    plt.figure(figsize = (10, 5))
    metric = metric.sort_values(by = ["metric"], ascending = False)
    metric.to_csv(out_f.split(".pdf")[0] + ".csv", sep = "\t", index = False)
    ax = sns.scatterplot(data = metric, y = "metric", x = "celltype", hue = "compartment", palette = color_map, s = 50, alpha = 0.5)
    
    # Plot baselines
    color_map = {"global": "gray"}
    for benchmark_model, benchmark_value in zip(benchmark_metric["benchmark"].tolist(), benchmark_metric["metric"].tolist()):
        print("Fraction of celltypes that outperform %s:" % benchmark_model, sum(metric["metric"] >= benchmark_value) / len(metric["celltype"].unique()), ("(%d out of %d)" % (sum(metric["metric"] >= benchmark_value), len(metric["celltype"].unique()))))
        if benchmark_model in color_map: model_color = color_map[benchmark_model]
        else:
            print("Please specify the color for benchmark %s. Using default color, gray, instead." % benchmark_model)
            model_color = "gray"
        plt.axhline(y = benchmark_value, color = model_color, label = benchmark_model, alpha = 1, linestyle = "--")
    
    plt.xlabel("Celltype")
    plt.ylabel(metric_name)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.ylim([-0.1, 1.02])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(bottom = False)
    plt.tight_layout()
    plt.savefig(out_f)
    plt.close()


def percentile_per_drug_target(selected_targets, model_outputs_df, evidence, out_f):
    color_map = {"Epithelial": "steelblue", "Stromal": "gold", "Endothelial": "darkorchid", "Immune": "firebrick", "Immune-Stromal": "darkorange", "Stromal-Epithelial": "yellowgreen", "Germ line": "violet"}
    
    for p in selected_targets:
        p_percs = model_outputs_df[model_outputs_df["name"] == p]
        if len(p_percs) == 0: return

        top_bottom_ranks = [p_percs.sort_values(by = "percentile", ascending = False).head(5)] + [p_percs.sort_values(by = "percentile", ascending = False).tail(5)]
        top_bottom_ranks = pd.concat(top_bottom_ranks)
        print(top_bottom_ranks)

        plt.figure(figsize = (6, 3))
        ax = sns.swarmplot(data = top_bottom_ranks, y = "celltype", x = "percentile", hue = "compartment", palette = color_map, alpha = 0.8)
        plt.xlim([0, 102])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend().set_visible(False)
        plt.tight_layout()
        plt.savefig(out_f + "_target-%s.pdf" % p)
        plt.close()


def main():
    
    # Disease of interest
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=str, default="all")
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--evidence', type=str, default=DATA_DIR)
    parser.add_argument('--benchmark_inventory_f', type=str, default="")
    parser.add_argument('--disease', type=str)
    parser.add_argument('--model_outputs_dir', type=str)
    parser.add_argument('--test_only', type=bool)
    parser.add_argument('--drug_targets', type=str, default="")
    args = parser.parse_args()

    # Seeds
    #seeds = list(range(1, 11)) # RA = 3, IBD = 5
    if args.seeds == "all": seeds = list(range(1, 11))
    else: seeds = [int(s) for s in args.seeds.split(",")]
    print(seeds)

    # Read meta graph
    metagraph = nx.read_edgelist(METAGRAPH_DIR, delimiter = "\t")
    
    # Read disease-drug evidence
    evidence = pd.read_csv(args.evidence + "tx_target/targets/disease_drug_evidence_%s.csv" % args.disease, sep = "\t")
    print(evidence)

    # Calculate performance
    metric_by_tissue_seeds = []
    metric_by_compartment_seeds = []
    benchmarks_metric_seeds = []
    benchmarks_roc_seeds = []
    
    for s in seeds:
        
        model_outputs_dir = args.model_outputs_dir + ("TS_seed=%s/" % str(s))
        save_prefix = "seed=%s" % str(s)

        model_outputs_df, test_proteins = read_model_data(model_outputs_dir, args.disease, args.test_only)
        utils.check_available_celltypes(metagraph, model_outputs_df)
        celltype2compartment, compartments, _, _ = utils.read_tissue_metadata(TS_TISSUE_DATA_DIR, "cell_ontology_class")

        # Calculate AP and ROC
        ap, roc, recall_k, precision_k, accuracy_k, ap_k = calculate_metrics(args.k, "celltype", test_proteins, model_outputs_df)
        metric = ap_k

        # Read benchmarks
        if args.benchmark_inventory_f != "":
            benchmarks = read_benchmarks(args.benchmark_inventory_f, args.disease, args.test_only, s)

            benchmark_ap, benchmark_roc, benchmark_recall_k, benchmark_precision_k, benchmark_accuracy_k, benchmark_ap_k = calculate_metrics(args.k, "benchmark", test_proteins, benchmarks)
            benchmark_metric = benchmark_ap_k
            benchmark_metric = pd.DataFrame.from_dict({"benchmark": list(benchmark_metric.keys()), "metric": list(benchmark_metric.values())})
            benchmarks_metric_seeds.append(benchmark_metric)

        # Aggregated targets across cell types
        if args.test_only:
            metric_by_compartment, celltype2compartment_df = utils.map_to_compartment(metric, celltype2compartment)
            metric_by_compartment_seeds.append(metric_by_compartment)

        # Individual targets
        if args.drug_targets != "":
            model_outputs_df = calculate_celltype_percentiles(model_outputs_df)
            celltype2compartment = {k: "-".join(v) for k, v in celltype2compartment.items()}
            model_outputs_df["compartment"] = model_outputs_df["celltype"].map(celltype2compartment)
        
            #selected_targets = ["JAK3", "IL6R"] # RA
            #selected_targets = ["ITGA4", "PPARG"] # IBD
            selected_targets = args.drug_targets.split(",")
            percentile_per_drug_target(selected_targets, model_outputs_df, evidence, "figures/%s_%s_percentiles" % (save_prefix, args.disease))

    if args.drug_targets == "" and len(seeds) > 1:
        metric_by_compartment_seeds = pd.concat(metric_by_compartment_seeds)
        if len(benchmarks_metric_seeds) > 0: benchmarks_metric_seeds = pd.concat(benchmarks_metric_seeds)
        plot_compartment_performance_across_seeds(metric_by_compartment_seeds, "Metric", benchmarks_metric_seeds, "figures/%s_metric_compartment_across_seeds.pdf" % args.disease)


if __name__ == "__main__":
    main()
