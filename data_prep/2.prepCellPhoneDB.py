# Prepare input files to CellPhoneDB v4 (https://github.com/ventolab/CellphoneDB)

# Outputs
#   - meta_CellPhoneDB.txt
#   - counts_CellPhoneDB.txt (optional; recommended to use h5ad)


import random
import argparse
import pandas as pd
from scipy.sparse import csr_matrix

from utils import read_ts_data


def get_meta(ts_data, out_f, groupby = "cell_ontology_class"):
    # Tab delimited, two columns = [cell ID, cell type label]
    meta = ts_data.obs[groupby].reset_index()
    print(meta.columns)
    meta.columns = ["Cell", "cell_type"]
    print(meta)
    meta.to_csv(out_f + "meta_CellPhoneDB.txt", sep = "\t", index = False)


def get_counts(ts_data, out_f):
    # Matrix where columns = cell ID, rows = gene counts, indices = gene ID
    print("The count matrix is...", ts_data.X.shape)
    counts = ts_data.X.toarray()
    print(counts)
    df = pd.DataFrame(data=counts)
    print(df)
    df.to_csv(out_f + "counts_CellPhoneDB.txt", sep = "\t", index = False)


def main():

    parser = argparse.ArgumentParser(description="Preparing input files for CellPhoneDB.")
    parser.add_argument("-data", type=str, default="/n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/data/TabulaSapiens.h5ad", help="Data as h5ad file.")
    parser.add_argument("-get_counts", type=bool, default=False, help="Convert sparse to dense matrix (memory intensive).")
    parser.add_argument("-subsample", type=bool, default=False, help="Randomly sample cells without replacement.")
    parser.add_argument("-percent", type=float, default=0, help="Percentage of cells to sample.")
    parser.add_argument("-output_subsample_f", type=str, help="Filename for subsampled cells.")
    parser.add_argument("-output_meta_f", type=str, default="/n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/processed/", help="Output prefix")
    args = parser.parse_args()
    
    print("Loading in Tabula Sapiens data...")
    ts_data = read_ts_data(args.data)
    print("Finished loading `ts_data`...")

    if args.subsample:
        num_cells = len(ts_data.obs)
        print("Number of cells:", num_cells)
        print("Sampling %.2f of cells:", num_cells * args.percent)
        random_idx = random.sample(range(num_cells), int(num_cells * args.percent))
        ts_data = ts_data[random_idx]
        print(ts_data)
        ts_data.write(args.output_subsample_f)

    print("Saving meta data...")
    get_meta(ts_data, args.output_meta_f)
    print("Finished saving meta data...")
    
    if args.get_counts:
        print("Converting count matrix to dense...")
        get_counts(ts_data, args.output_meta_f)
        print("Finished converting count matrix to dense...")
    
    print("All finished!")


if __name__ == "__main__":
    main() 
