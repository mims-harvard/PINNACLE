from utils import read_ts_data

TABULA_SAPIENS_DIR = "../data/raw/TabulaSapiens.h5ad"
OUTPUT_DIR = "../data/single_cell/"


ts_data = read_ts_data(TABULA_SAPIENS_DIR)
print(ts_data)
print(ts_data.obs[["anatomical_information", "organ_tissue", "compartment", "cell_ontology_class"]])
ts_data.obs.to_csv(OUTPUT_DIR + "ts_data_tissue.csv", sep = "\t")