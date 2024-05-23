from utils import read_ts_data

import sys
sys.path.insert(0, '..') # add data_config to path
from data_config import TABULA_SAPIENS_DIR, TS_TISSUE_DATA_DIR


ts_data = read_ts_data(TABULA_SAPIENS_DIR)
print(ts_data)
print(ts_data.obs[["anatomical_information", "organ_tissue", "compartment", "cell_ontology_class"]])
ts_data.obs.to_csv(TS_TISSUE_DATA_DIR, sep = "\t")