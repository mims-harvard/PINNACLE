from utils import read_ts_data

TABULA_SAPIENS_DIR = "/n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/data/TabulaSapiens.h5ad"
OUTPUT_DIR = "/n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/processed/"


ts_data = read_ts_data(TABULA_SAPIENS_DIR)
print(ts_data)
print(ts_data.obs[["anatomical_information", "organ_tissue", "compartment", "cell_ontology_class"]])
ts_data.obs.to_csv(OUTPUT_DIR + "ts_data_tissue.csv", sep = "\t")