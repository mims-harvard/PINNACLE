from utils import read_ts_data

from data_config import TABULA_SAPIENS_DIR, OUTPUT_DIR


ts_data = read_ts_data(TABULA_SAPIENS_DIR)
print(ts_data)
print(ts_data.obs[["anatomical_information", "organ_tissue", "compartment", "cell_ontology_class"]])
ts_data.obs.to_csv(OUTPUT_DIR + "ts_data_tissue.csv", sep = "\t")