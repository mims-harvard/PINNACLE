############################################################
# DATA DIRECTORY

DATA_DIR = "../data/"                                       # Data directory
RAW_DATA_DIR = DATA_DIR + "raw/"                            # Directory with all draw data
OUTPUT_DIR = DATA_DIR + "networks/"                         # Directory with all networks


############################################################
# DATA FILES

# Single cell transcriptomic atlas
TABULA_SAPIENS_DIR = RAW_DATA_DIR + "TabulaSapiens.h5ad"    # Single-cell atlas
TS_TISSUE_DATA_DIR = OUTPUT_DIR + "ts_data_tissue.csv"      # Generated data_prep/presave_compartments.py

# Global PPI network
PPI_DIR = OUTPUT_DIR + "global_ppi_edgelist.txt"            # Global PPI network
METAGRAPH_DIR = OUTPUT_DIR + "mg_edgelist.txt"              # Metagraph

############################################################
# AUXILIARY DATA FILES

CELL_ONTOLOGY_DIR = DATA_DIR + "cell-ontology/cl-full.obo"  # Cell ontology
BTO_DIR = RAW_DATA_DIR + "BTO.obo"                          # Tissue ontology