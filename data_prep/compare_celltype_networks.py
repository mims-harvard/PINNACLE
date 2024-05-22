# Compare pairs of cell type networks
import argparse
import numpy as np

from utils import load_global_PPI, load_celltype_ppi, count_cells_per_celltype
from utils import jaccard_similarity

import sys
sys.path.insert(0, '..') # add data_config to path
from data_config import PPI_DIR, OUTPUT_DIR


def overlapping_celltypes(celltype_ppi_1, celltype_ppi_2, cells_per_celltype):
    celltypes_1 = set(list(celltype_ppi_1.keys())) #set([c[1] for c in celltype_ppi_1])
    celltypes_2 = set(list(celltype_ppi_2.keys())) #set([c[1] for c in celltype_ppi_2])

    overlap = celltypes_1.intersection(celltypes_2)
    only_in_1 = celltypes_1.difference(celltypes_2)
    only_in_2 = celltypes_2.difference(celltypes_1)

    print("Overlapping cell types:", len(overlap), overlap)
    print("Cell types in 1 but not 2:", len(only_in_1), only_in_1)

    # Question: If there are cell types that are not overlapping, is itm because
    #           of "rare" cell types? (e.g., small number of cells)
    print(">=10K cells", [c for c in only_in_1 if cells_per_celltype[c] >= 10000])
    print(">=5K cells", [c for c in only_in_1 if cells_per_celltype[c] >= 5000])
    print("<1K cells", [c for c in only_in_1 if cells_per_celltype[c] < 1000])
    print("<100 cells", [c for c in only_in_1 if cells_per_celltype[c] < 100])

    print("Cell types in 2 but not 1:", len(only_in_2), only_in_2)

    return overlap, only_in_1, only_in_2


def jaccard_sims_comparisons(celltype_ppi_1, celltype_ppi_2, cells_per_celltype, global_ppi):
    sims_over10K = []
    sims_lessthan1K = []
    sims_lessthan500 = []

    sims_equal100 = []
    sims_equal200 = []
    sims_equal300 = []

    global_sims_1_over10K = []
    global_sims_1_lessthan1K = []
    global_sims_1_lessthan500 = []

    global_sims_2_over10K = []
    global_sims_2_lessthan1K = []
    global_sims_2_lessthan500 = []

    global_sims_1_equal100 = []
    global_sims_1_equal200 = []
    global_sims_1_equal300 = []

    global_sims_2_equal100 = []
    global_sims_2_equal200 = []
    global_sims_2_equal300 = []
    for c, c_count in cells_per_celltype.items():
        if c in celltype_ppi_1 and c in celltype_ppi_2:
            sims = jaccard_similarity(set(celltype_ppi_1[c]), set(celltype_ppi_2[c]))
            global_sims_1 = jaccard_similarity(set(celltype_ppi_1[c]), global_ppi)
            global_sims_2 = jaccard_similarity(set(celltype_ppi_2[c]), global_ppi)

        else:
            continue
        if c_count >= 10000:
            sims_over10K.append(sims)
            global_sims_1_over10K.append(global_sims_1)
            global_sims_2_over10K.append(global_sims_2)
        if c_count < 1000:
            sims_lessthan1K.append(sims)
            global_sims_1_lessthan1K.append(global_sims_1)
            global_sims_2_lessthan1K.append(global_sims_2)
        if c_count < 500:
            sims_lessthan500.append(sims)
            global_sims_1_lessthan500.append(global_sims_1)
            global_sims_2_lessthan500.append(global_sims_2)
        if c_count == 100:
            sims_equal100.append(sims)
            global_sims_1_equal100.append(global_sims_1)
            global_sims_2_equal100.append(global_sims_2)
        if c_count == 200:
            sims_equal200.append(sims)
            global_sims_1_equal200.append(global_sims_1)
            global_sims_2_equal200.append(global_sims_2)
        if c_count == 300:
            sims_equal300.append(sims)
            global_sims_1_equal300.append(global_sims_1)
            global_sims_2_equal300.append(global_sims_2)

    print("Jaccard similarity for cell types >= 10K:", "Min: %.2f, Max: %.2f, Average: %.2f +/- %.2f" % (min(sims_over10K), max(sims_over10K), np.mean(sims_over10K), np.std(sims_over10K)))
    print("Jaccard similarity for cell types < 1K:", "Min: %.2f, Max: %.2f, Average: %.2f +/- %.2f" % (min(sims_lessthan1K), max(sims_lessthan1K), np.mean(sims_lessthan1K), np.std(sims_lessthan1K)))
    print("Jaccard similarity for cell types < 500:", "Min: %.2f, Max: %.2f, Average: %.2f +/- %.2f" % (min(sims_lessthan500), max(sims_lessthan500), np.mean(sims_lessthan500), np.std(sims_lessthan500)))
    if len(sims_equal100) > 0: print("Jaccard similarity for cell types = 100:", "Min: %.2f, Max: %.2f, Average: %.2f +/- %.2f" % (min(sims_equal100), max(sims_equal100), np.mean(sims_equal100), np.std(sims_equal100)))
    if len(sims_equal200) > 0: print("Jaccard similarity for cell types = 200:", "Min: %.2f, Max: %.2f, Average: %.2f +/- %.2f" % (min(sims_equal200), max(sims_equal200), np.mean(sims_equal200), np.std(sims_equal200)))
    if len(sims_equal300) > 0: print("Jaccard similarity for cell types = 300:", "Min: %.2f, Max: %.2f, Average: %.2f +/- %.2f" % (min(sims_equal300), max(sims_equal300), np.mean(sims_equal300), np.std(sims_equal300)))

    print("Jaccard similarity of global & celltype1 for cell types >= 10K:", "Min: %.2f, Max: %.2f, Average: %.2f +/- %.2f" % (min(global_sims_1_over10K), max(global_sims_1_over10K), np.mean(global_sims_1_over10K), np.std(global_sims_1_over10K)))
    print("Jaccard similarity of global & celltype1 for cell types < 1K:", "Min: %.2f, Max: %.2f, Average: %.2f +/- %.2f" % (min(global_sims_1_lessthan1K), max(global_sims_1_lessthan1K), np.mean(global_sims_1_lessthan1K), np.std(global_sims_1_lessthan1K)))
    print("Jaccard similarity of global & celltype1 for cell types < 500:", "Min: %.2f, Max: %.2f, Average: %.2f +/- %.2f" % (min(global_sims_1_lessthan500), max(global_sims_1_lessthan500), np.mean(global_sims_1_lessthan500), np.std(global_sims_1_lessthan500)))
    if len(global_sims_1_equal100) > 0: print("Jaccard similarity of global & celltype1 for cell types = 100:", "Min: %.2f, Max: %.2f, Average: %.2f +/- %.2f" % (min(global_sims_1_equal100), max(global_sims_1_equal100), np.mean(global_sims_1_equal100), np.std(global_sims_1_equal100)))
    if len(global_sims_1_equal200) > 0: print("Jaccard similarity of global & celltype1 for cell types = 200:", "Min: %.2f, Max: %.2f, Average: %.2f +/- %.2f" % (min(global_sims_1_equal200), max(global_sims_1_equal200), np.mean(global_sims_1_equal200), np.std(global_sims_1_equal200)))
    if len(global_sims_1_equal300) > 0: print("Jaccard similarity of global & celltype1 for cell types = 300:", "Min: %.2f, Max: %.2f, Average: %.2f +/- %.2f" % (min(global_sims_1_equal300), max(global_sims_1_equal300), np.mean(global_sims_1_equal300), np.std(global_sims_1_equal300)))

    print("Jaccard similarity of global & celltype2 for cell types >= 10K:", "Min: %.2f, Max: %.2f, Average: %.2f +/- %.2f" % (min(global_sims_2_over10K), max(global_sims_2_over10K), np.mean(global_sims_2_over10K), np.std(global_sims_2_over10K)))
    print("Jaccard similarity of global & celltype2 for cell types < 1K:", "Min: %.2f, Max: %.2f, Average: %.2f +/- %.2f" % (min(global_sims_2_lessthan1K), max(global_sims_2_lessthan1K), np.mean(global_sims_2_lessthan1K), np.std(global_sims_2_lessthan1K)))
    print("Jaccard similarity of global & celltype2 for cell types < 500:", "Min: %.2f, Max: %.2f, Average: %.2f +/- %.2f" % (min(global_sims_2_lessthan500), max(global_sims_2_lessthan500), np.mean(global_sims_2_lessthan500), np.std(global_sims_2_lessthan500)))
    if len(global_sims_2_equal100) > 0: print("Jaccard similarity of global & celltype2 for cell types = 100:", "Min: %.2f, Max: %.2f, Average: %.2f +/- %.2f" % (min(global_sims_2_equal100), max(global_sims_2_equal100), np.mean(global_sims_2_equal100), np.std(global_sims_2_equal100)))
    if len(global_sims_2_equal200) > 0: print("Jaccard similarity of global & celltype2 for cell types = 200:", "Min: %.2f, Max: %.2f, Average: %.2f +/- %.2f" % (min(global_sims_2_equal200), max(global_sims_2_equal200), np.mean(global_sims_2_equal200), np.std(global_sims_2_equal200)))
    if len(global_sims_2_equal300) > 0: print("Jaccard similarity of global & celltype2 for cell types = 300:", "Min: %.2f, Max: %.2f, Average: %.2f +/- %.2f" % (min(global_sims_2_equal300), max(global_sims_2_equal300), np.mean(global_sims_2_equal300), np.std(global_sims_2_equal300)))


def main():
    parser = argparse.ArgumentParser(description="Constructing meta graph.")
    parser.add_argument("-celltype_ppi_1", type=str, help="Filename (prefix) of cell type PPI 1.")
    parser.add_argument("-celltype_ppi_2", type=str, help="Filename (prefix) of cell type PPI 2.")
    args = parser.parse_args()

    """
    python compare_celltype_networks.py -celltype_ppi_1 /n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/processed/all/ppi_TabulaSapiens_maxpval=1.0.csv -celltype_ppi_2 /n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/processed/subsample100/ppi_TabulaSapiens_subsample_maxpval=1.0.csv
    python compare_celltype_networks.py -celltype_ppi_1 /n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/processed/all/ppi_TabulaSapiens_maxpval=1.0.csv -celltype_ppi_2 /n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/processed/subsample200/ppi_TabulaSapiens_subsample_maxpval=1.0.csv
    python compare_celltype_networks.py -celltype_ppi_1 /n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/processed/all/ppi_TabulaSapiens_maxpval=1.0.csv -celltype_ppi_2 /n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/processed/subsample300/ppi_TabulaSapiens_subsample_maxpval=1.0.csv
    """

    global_ppi = load_global_PPI(PPI_DIR)

    celltype_ppi_1 = load_celltype_ppi(args.celltype_ppi_1)
    celltype_ppi_1 = {k[1]: v for k, v in celltype_ppi_1.items()}
    celltype_ppi_2 = load_celltype_ppi(args.celltype_ppi_2)
    celltype_ppi_2 = {k[1]: v for k, v in celltype_ppi_2.items()}
    cells_per_celltype = count_cells_per_celltype(OUTPUT_DIR + "ts_data_tissue.csv")
    print(cells_per_celltype)

    print("Number of cell types in PPI 1:", len(celltype_ppi_1))
    print("Number of cell types in PPI 2:", len(celltype_ppi_2))

    overlap, only_in_1, only_in_2 = overlapping_celltypes(celltype_ppi_1, celltype_ppi_2, cells_per_celltype)

    jaccard_sims_comparisons(celltype_ppi_1, celltype_ppi_2, cells_per_celltype, list(global_ppi.nodes))
    



if __name__ == "__main__":
    main() 
