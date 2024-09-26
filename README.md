# PINNACLE: Contextual AI models for single-cell protein biology

**Authors**:
- [Michelle M. Li](http://michellemli.com)
- [Yepeng Huang](http://zitniklab.hms.harvard.edu)
- [Marissa Sumathipala](http://zitniklab.hms.harvard.edu)
- [Man Qing Liang](http://zitniklab.hms.harvard.edu)
- [Alberto Valdeolivas]()
- [Ashwin Ananthakrishnan]()
- [Katherine Liao]()
- [Daniel Marbach]()
- [Marinka Zitnik](http://zitniklab.hms.harvard.edu)

## Overview of PINNACLE

Protein interaction networks are a critical component in studying the function and therapeutic potential of proteins. However, accurately modeling protein interactions across diverse biological contexts, such as tissues and cell types, remains a significant challenge for existing algorithms.

We introduce PINNACLE, a flexible geometric deep learning approach that trains on contextualized protein interaction networks to generate context-aware protein representations. Leveraging a multi-organ single-cell transcriptomic atlas of humans, PINNACLE provides 394,760 protein representations split across 156 cell-type contexts from 24 tissues and organs. We demonstrate that PINNACLE's contextualized representations of proteins reflect cellular and tissue organization and PINNACLE's tissue representations enable zero-shot retrieval of tissue hierarchy. Infused with cellular and tissue organization, our contextualized protein representations can easily be adapted for diverse downstream tasks.

We fine-tune PINNACLE to study the genomic effects of drugs in multiple cellular contexts and show that our context-aware model significantly outperforms state-of-the-art, yet context-agnostic, models. Enabled by our context-aware modeling of proteins, PINNACLE is able to nominate promising protein targets and cell-type contexts for further investigation. PINNACLE exemplifies and empowers the long-standing paradigm of incorporating context-specific effects for studying biological systems, especially the impact of disease and therapeutics.

### The PINNACLE Algorithm

PINNACLE is a self-supervised geometric deep learning model that can generate protein representations in diverse celltype contexts. It is trained on a set of context-aware protein interaction networks unified by a cellular and tissue network to produce contextualized protein representations based cell type activation. Unlike existing approaches, which do not consider biological context, PINNACLE produces multiple representations of proteins based on their cell type context, representations of the cell type contexts themselves, and representations of the tissue hierarchy. 

Given the multi-scale nature of the model inputs, PINNACLE is equipped to learn the topology of proteins, cell types, and tissues in a single unified embedding space. PINNACLE uses protein-, cell type-, and tissue-level attention mechanisms and objective functions to inject cellular and tissue organization into the embedding space. Intuitively, pairs of nodes that share an edge should be embedded nearby, proteins of the same cell type context should be embedded nearby (and far from proteins in other cell type contexts), and proteins should be embedded close to their cell type context (and far from other cell type contexts).

<p align="center">
<img src="img/pinnacle_overview.png?raw=true" width="700" >
</p>


## Installation and Setup

### :one: Download the Repo

First, clone the GitHub repository:

```
git clone https://github.com/mims-harvard/PINNACLE
cd PINNACLE
```

### :two: Set Up Environment

This codebase leverages Python, Pytorch, Pytorch Geometric, etc. To create an environment with all of the required packages, please ensure that [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) is installed and then execute the commands:

```
conda env create -f environment.yml
conda activate pinnacle
bash install_pyg.sh
```

### :three: Download Datasets

The data is hosted on [Figshare](https://figshare.com/articles/software/PINNACLE/22708126). To maintain the directory structure while downloading the files, make sure to select all files and download in the original format. Make sure to also unzip all files in the download.

We provide the following datasets for training PINNACLE:
- Global reference protein interaction network
- Cell type specific protein interaction networks
- Metagraph of cell type and tissue relationships

The networks are provided in the appropriate format for PINNACLE. If you would like to use your own set of contextualized networks, please adhere to the format used in the cell type specific protein interaction networks (see [README](https://github.com/mims-harvard/PINNACLE/blob/main/data_prep/README.md) in `data_prep` folder for more details). The file should be structured as a tab-delimited table, where each line contains information for a single context. Each line must contain the following elements (in this order): index, context name (e.g., cell type name), comma-delimited list of nodes. The lists of nodes are used to extract a subgraph from the global reference network (e.g., global reference protein interaction network).

### :four: (Optional) Download Model Checkpoints
We also provide checkpoints for PINNACLE after pretraining. The checkpoints for PINNACLE can be found [here](https://figshare.com/articles/software/PINNACLE/22708126). Make sure all downloaded files are unzipped. You can use these checkpoints (and/or embeddings) directly with the scripts in the `finetune_pinnacle` folder instead of training the models yourself.

## Usage

### Finetune PINNACLE on Your Own Datasets

You can finetune PINNACLE on your own datasets by using our provided model checkpoints or contextualized representations (i.e., no re-training needed). Please review this [README](https://github.com/mims-harvard/PINNACLE/blob/main/finetune_pinnacle/README.md) to learn how to preprocess and finetune PINNACLE on your own datasets.

### Train PINNACLE

You can reproduce our results or pretrain PINNACLE on your own networks:
```
cd pinnacle
python train.py \
        --G_f ../data/networks/global_ppi_edgelist.txt \
        --ppi_dir ../data/networks/ppi_edgelists/ \
        --mg_f ../data/networks/mg_edgelist.txt \
        --save_prefix ../data/pinnacle_embeds/
```

To see and/or modify the default hyperparameters, please see the `get_hparams()` function in `pinnacle/parse_args.py`.

An example bash script is provided in `pinnacle/run_pinnacle.sh`.

### Visualize PINNACLE Representations

After training PINNACLE, you can visualize PINNACLE's representations using `evaluate/visualize_representations.py`.

### Finetune PINNACLE for nominating therapeutic targets

After training PINNACLE (you may also simply use our already-trained models), you can finetune PINNACLE for any downstream biomedical task of interest. Here, we provide instructions for nominating therapeutic targets. An example bash script can be found [here](https://github.com/mims-harvard/PINNACLE/blob/main/finetune_pinnacle/run_model.sh).

:sparkles: To finetune PINNACLE for nominating therapeutic targets of rheumatoid arthritis:

```
cd finetune_pinnacle
python train.py \
        --disease EFO_0000685 \
        --embeddings_dir ./data/pinnacle_embeds/
```

:sparkles: To finetune PINNACLE for nominating therapeutic targets of inflammatory bowel disease:

```
cd finetune_pinnacle
python train.py \
        --disease EFO_0003767 \
        --embeddings_dir ./data/pinnacle_embeds/
```

To generate predictions on a different therapeutic area, simply find the disease ID from OpenTargets and change the `---disease` flag.

To see and/or modify the default hyperparameters, please see the `get_hparams()` function in `finetune_pinnacle/train_utils.py`.

## Additional Resources

- [Paper](https://www.biorxiv.org/content/10.1101/2023.07.18.549602)
- [Demo](https://huggingface.co/spaces/michellemli/PINNACLE/)
- [Project Website](https://zitniklab.hms.harvard.edu/projects/PINNACLE/)

```
@article{pinnacle,
  title={Contextual AI models for single-cell protein biology},
  author={Li, Michelle M and Huang, Yepeng and Sumathipala, Marissa and Liang, Man Qing and Valdeolivas, Alberto and Ananthakrishnan, Ashwin N and Liao, Katherine and Marbach, Daniel and Zitnik, Marinka},
  journal={Nature Methods},
  pages={1--12},
  year={2024},
  publisher={Nature Publishing Group US New York}
}
```


## Questions

Please leave a Github issue or contact Michelle Li at michelleli@g.harvard.edu.
