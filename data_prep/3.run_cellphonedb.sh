#!/bin/bash

# conda activate cpdb

cellphonedb method statistical_analysis ../data/single_cell/cpdb/ranked_TabulaSapiens_subsample_${1}_meta_CellPhoneDB.txt ../data/single_cell/cpdb/ranked_TabulaSapiens_subsample_${1}.h5ad --counts-data hgnc_symbol --output-path ../data/single_cell/cpdb/CellPhoneDB_results_ranked_TabulaSapiens_subsample_${1} --subsampling --subsampling-log false
