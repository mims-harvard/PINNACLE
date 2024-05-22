import pandas as pd


def read_tissue_metadata(f, annotation):
    tissue_metadata = pd.read_csv(f, sep ="\t")
    print(tissue_metadata)

    celltype2compartment = dict()
    celltype2tissue = dict()
    for c in tissue_metadata[annotation].unique():
        c_compartment = tissue_metadata[tissue_metadata[annotation] == c]["compartment"].unique()
        c_compartment = [i.capitalize() for i in c_compartment]

        c_tissue = tissue_metadata[tissue_metadata[annotation] == c]["organ_tissue"].unique()
        c_tissue = [" ".join(t.split("_")) for t in c_tissue]

        if c in celltype2compartment: 
            assert celltype2compartment[c] == c_compartment, c
        if c in celltype2tissue: 
            assert celltype2tissue[c] == c_tissue, c

        celltype2compartment[c] = c_compartment
        celltype2tissue[c] = c_tissue

    return celltype2compartment, tissue_metadata["compartment"].unique(), celltype2tissue, tissue_metadata["organ_tissue"].unique()


def filter_model_data(df, test_only):
    if test_only: df = df[df["type"] == "test"] # Keep only test for evaluation
    else: df = df[df["type"] != "train"] # Remove train for evaluation
    df = df[df["y"] >= 0]
    return df


def map_to_compartment(metric, celltype2compartment):
    celltype2compartment = {k: "-".join(v) for k, v in celltype2compartment.items()}

    metric_df = pd.DataFrame.from_dict({"celltype": list(metric.keys()), "metric": list(metric.values())})
    celltype2compartment_df = pd.DataFrame.from_dict({"celltype": list(celltype2compartment.keys()), "compartment": list(celltype2compartment.values())})

    metric_by_compartment = metric_df.merge(celltype2compartment_df)
    assert len(metric_by_compartment) == len(metric), (len(metric_by_compartment), len(metric))

    return metric_by_compartment, celltype2compartment_df


def get_celltype2tissue(metagraph, bto_names):
    celltype2tissue = dict()
    for n in metagraph.nodes():
        if n.startswith("BTO"): continue
        tissue_neighbors = [t for t in metagraph.neighbors(n) if t.startswith("BTO")]
        if len(tissue_neighbors) == 1: celltype2tissue[n] = bto_names[tissue_neighbors[0]]
        else: celltype2tissue[n] = ";".join([bto_names[t] for t in tissue_neighbors])
    return celltype2tissue




####################################################################################################
#
# SANITY CHECKS
#
####################################################################################################


def check_no_leakage_protein_split(sanity_check):
    for p in sanity_check["name"].unique():
        p_df = sanity_check[sanity_check["name"] == p]
        assert len(p_df["type"].unique()) == 1, (p, p_df["type"].unique())


def check_available_celltypes(celltypes, model_outputs_df):
    model_celltypes = model_outputs_df["celltype"].unique()
    assert len(set(model_celltypes).intersection(set(celltypes))) == len(set(model_celltypes))