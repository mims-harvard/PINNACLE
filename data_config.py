############################################################
# DATA DIRECTORY

DATA_DIR = "../data/"
RAW_DATA_DIR = DATA_DIR + "raw/"
OUTPUT_DIR = DATA_DIR + "networks/"


############################################################
# DATA FILES

# Single cell transcriptomic atlas
TABULA_SAPIENS_DIR = RAW_DATA_DIR + "TabulaSapiens.h5ad"
TS_TISSUE_DATA_DIR = RAW_DATA_DIR + "ts_data_tissue.csv"

# Global PPI network
PPI_DIR = OUTPUT_DIR + "global_ppi_edgelist.txt"
METAGRAPH_DIR = OUTPUT_DIR + "mg_edgelist.txt"

############################################################
# AUXILIARY DATA FILES

CELL_ONTOLOGY_DIR = DATA_DIR + "cell-ontology/cl-full.obo"
BTO_DIR = RAW_DATA_DIR + "BTO.obo"