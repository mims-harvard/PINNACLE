#!/bin/bash
#SBATCH -c 5                               
#SBATCH -t 0-08:00                         # Runtime in D-HH:MM format
#SBATCH -p short                           # Partition to run in
#SBATCH --mem=100G                         # Memory total in MB (for all cores)
#SBATCH -o cellphonedb.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e cellphonedb.err                 # File to which STDERR will be written, including job ID (%j)

# conda activate cpdb


# Without subsampling
# cellphonedb method statistical_analysis /n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/processed/meta_CellPhoneDB.txt /n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/data/TabulaSapiens.h5ad --output-path /n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/processed/CellPhoneDB_results



# With subsampling
# cellphonedb method statistical_analysis /n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/processed/ranked_TabulaSapiens_subsample_0_meta_CellPhoneDB.txt /n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/processed/ranked_TabulaSapiens_subsample_0.h5ad --counts-data hgnc_symbol --output-path /n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/processed/CellPhoneDB_results_ranked_TabulaSapiens_subsample_0
# cellphonedb method statistical_analysis /n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/processed/subsample200/ranked_TabulaSapiens_subsample_${1}_meta_CellPhoneDB.txt /n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/processed/subsample200/ranked_TabulaSapiens_subsample_${1}.h5ad --counts-data hgnc_symbol --output-path /n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/processed/subsample200/CellPhoneDB_results_ranked_TabulaSapiens_subsample_${1}

cellphonedb method statistical_analysis /n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/processed/all/cpdb/ranked_TabulaSapiens_subsample_${1}_meta_CellPhoneDB.txt /n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/processed/all/cpdb/ranked_TabulaSapiens_subsample_${1}.h5ad --counts-data hgnc_symbol --output-path /n/data1/hms/dbmi/zitnik/lab/datasets/2022-09-TabulaSapiens/processed/all/cpdb/CellPhoneDB_results_ranked_TabulaSapiens_subsample_${1} --subsampling --subsampling-log false
