# Constructing cell type PPI from Tabula Sapiens

## Step 0: Set up environment & data variables

This codebase leverages Python, NetworkX, Scanpy, etc. To create an environment with all of the required packages, please ensure that [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) is installed and then execute the commands:

```
conda env create -f scRNA_env.yml
conda activate scRNA_env
```

The data variables for the single cell transcriptomic atlas, global PPI network, auxiliary data, and output directory are in `data_config.py`. Be sure to update them accordingly.


## Step 1: Rank and extract genes

Required environment for each script used:
- `0.constructPPI.py` depends on `conda activate scRNA_env`

### Detailed instructions

First, run `0.constructPPI.py` to construct cell type specific PPI networks. It is possible to split the task into two: (1) Ranking the genes in each cell type, and (2) extracting the cell type specific PPI network from the global PPI network based on the rankings.

To rank genes:

```python 0.constructPPI.py -rank True -rank_pval_filename PATH/TO/OUTPUT```

The outputs of this command are:
- AnnData object with ranked genes information (`.h5ad` file type)
- Table (`.csv`, tab-delimited) of the ranked genes and their corresponding p-values (rows) for each cell type (columns)

To extract genes:

```python 0.constructPPI.py -rank False -rank_pval_filename PATH/TO/INPUT -celltype_ppi_filename PATH/TO/OUTPUT -max_pval 1 -max_num_genes 4000```

The outputs of this command are:
- Table (`.csv`, tab-delimited) of the cell types (two columns: one for index of cell type, one for name of cell type) and their genes (one column, comma-delimited string of proteins)

How to use each flag:
- Use the `-rank` flag to indicate whether you want to rank the genes (`True`) or extract the PPI network (`False`).
- Indicate the column of annotations (in the input AnnData) using the `-annotation` flag.
- To subsample equal number of cells per cell type, set the `-subsample` flag to `True` and indicate the number of iterations using the `-iterations`.
- When `-rank False`, the relevant flags to include are `-max_pval` (float, default is `1`) and `-max_num_genes` (integer, default is `3000`).
- Provide the path to the cell type PPI network using the `-celltype_ppi_filename` flag.

Important parameters that are NOT flags:
- The global `TABULA_SAPIENS_DIR` variable holds the directory to AnnData object (`.h5ad` file type). The AnnData must already have cell type anotations. The default column is `cell_ontology_class`. Be sure to update `TABULA_SAPIENS_DIR` with the path to your desired input data.
- Cell types with fewer than `num_cells_cutoff` cells are excluded. By default, `num_cells_cutoff = 100`.


## Step 2: Evaluate cell type specific PPI networks

Required environment for each script used:
- `1.evaluatePPI.py` depends on `conda activate scRNA_env`

### Detailed instructions

Next, run `1.evaluatePPI.py` to evaluate the cell type specific PPI networks.

```python 1.evaluatePPI.py -celltype_ppi PATH/TO/INPUT```

The outputs of this command are:
- PDFs of figures produced during evaluation

How to use each flag:
- Provide the path to the cell type PPI networks (output from the previous step) using the `-celltype_ppi` flag
- Indicate the appropriate `-max_pval` (float, default is `1`) and `-max_num_genes` (integer, default is `4000`). These must be the same values as those used to run `0.constructPPI.py`.


## Step 3: Run CellPhoneDB for cell-cell interactions

Required environment for each script used:
- `2.prepCellPhoneDB.py` depends on `conda activate scRNA_env`
- `3.run_cellphonedb.sh` depends on `conda activate cpdb`
    1. `conda create -n cpdb python=3.7`
    2. `conda activate cpdb`
    3. `pip install cellphonedb`

### Detailed instructions

Run `2.prepCellPhoneDB.py` to prepare inputs for CellPhoneDB.

```python 2.prepCellPhoneDB.py -data PATH/TO/H5AD/DATA -output PATH/TO/OUTPUT```

The script outputs are:
- Meta data (tab-delimited table with two columns: first column is the cell ID, second column is the cell type annotation).
- (Optional) Count matrix (tab-delimited, columns are cells and rows are genes).

How to use each flag:
- Use the `-data` flag to provide the directory of the AnnData object (`.h5ad` file type). The AnnData must already have cell type anotations. The default column is `cell_ontology_class`.
- For the count matrix, CellPhoneDB can either take `.h5ad` (recommended) or a `.txt` with the dense matrix. By default, the script does NOT create a `.txt` file. But if that is desired, set `-get_counts True`.
- Provide an output directory to save the files produced by the script using the `-output` flag.

To run CellPhoneDB, see `3.run_cellphonedb.sh`:

```cellphonedb method statistical_analysis meta_CellPhoneDB.txt ranked_TabulaSapiens.h5ad --counts-data hgnc_symbol --output-path CellPhoneDB_results```

The outputs of this command are:
- See CellPhoneDB documentation for most up to date information
- The `means.txt` file contains mean values for each ligand-receptor interaction. Each row is cell pair interaction.
- The `significant_means.txt` file (similar structure as `means.txt`) contains significant means for all interacting partners. If `p-value < 0.05`, the value will be the mean. Alternatively, the value is set to 0. This file also contains the `rank`, which is the total number of significant p-values for each interaction divided by the number of cell type to cell type comparisons.
- The `pvalues.txt` file (similar structure as `means.txt`) contains p-values for the all the interacting partners. THe p-value refers to the enrichment of the interacting ligand-receptor pair in each of the interacting pairs of cell types.
- The `deconvoluted.txt` file gives additional information for each of the interacting partners. This is important as some of the interacting partners are heteromers. In other words, multiple molecules have to be expressed in the same cluster in order for the interacting partner to be functional.


## Step 4: Construct cell-cell interaction network

Required environment for each script used:
- `4.constructCCI.py` depends on `conda activate scRNA_env`

### Detailed instructions

Run `4.constructCCI.py` to summarize CellPhoneDB outputs and construct a cell-cell interaction network.

```python 4.constructCCI.py -cpdb_output PATH/TO/CELLPHONEDB/pvalues.txt -cci_edgelist PATH/TO/OUTPUT```

The script outputs are:
- Edgelist of cell-cell interaction network (tab-delimited)

How to use each flag:
- Provide the directory to the CellPhoneDB outputs using the `-cpdb_output` flag. This script will automatically extract all `pvalues.txt` files from the directory
- Provide the filename for the cell-cell interaction network edgelist using `-cci_edgelist` flag.


## Step 5: Construct meta graph

Required environment for each script used:
- `5.constructMG.py` depends on `conda activate scRNA_env`

### Detailed instructions

Run `5.constructMG.py` to extract tissue-tissue associations and combine all components of the meta graph.

```python 5.constructMG.py -celltype_ppi PATH/TO/CELLTYPE/PPI -cci_edgelist PATH/TO/CCI/EDGELIST -mg_edgelist PATH/TO/OUTPUT```

The script outputs are:
- Edgelist of meta graph (tab-delimited)

How to use each flag:
- Provide the path to the cell type PPI networks using the `-celltype_ppi` flag.
- Provide the filename for the cell-cell interaction network edgelist using `-cci_edgelist` flag.
- Provide the filename for the meta graph edgelist using `-mg_edgelist` flag.