# Finetuning PINNACLE

You can finetune PINNACLE on your own datasets by using our provided model checkpoints or contextualized representations (i.e., no re-training needed).

## Step-by-Step Instructions

We provide detailed instructions for fine-tuning PINNACLE on the pretrained contextualized protein representations.

### Step 1: Curate fine-tuning data

You may use `prepare_txdata.py` as an example.

The required outputs of your script are:
- Positive proteins (dict)
    - Filename: `positive_proteins_<task_name>.json`
    - Data structure: `{"<celltype name>": ["<protein name>"]}`
- Negative proteins (dict)
    - Filename: `negative_proteins_<task_name>.json`
    - Data structure: `{"<celltype name>": ["<protein name>"]}`
- Raw data (list)
    - Filename: `raw_data_<task_name>.json`
    - Data structure: `["<protein name>"]`

### Step 2: Split and format data

With the three files created in Step 1, run `data_prep.py`. The outputs of this script are:
- Data split indices (dict)
    - Filename: `data_split_<task_name>.json`
    - Data Structure: `{"pos_train_indices": [...], "neg_train_indices": [...], "pos_test_indices": [...], "neg_test_indices": [...]}`
- Data split name (dict)
    - Filename: `data_split_<task_name>_name.json`
    - Data Structure: `{"pos_train_names": [...], "neg_train_names": [...], "pos_test_names": [...], "neg_test_names": [...]}`

Example command:
```
python data_prep.py \
    --embeddings_dir ../data/pinnacle_embeds/ \
    --embed pinnacle \
    --celltype_ppi ../data/networks/ \
    --positive_proteins_prefix ../data/therapeutic_target_task/positive_proteins_EFO_0000685 \
    --negative_proteins_prefix ../data/therapeutic_target_task/negative_proteins_EFO_0000685 \
    --raw_data_prefix ../data/therapeutic_target_task/raw_targets_EFO_0000685 \
    --data_split_path ../data/therapeutic_target_task/data_split_EFO_0000685
```

### Step 3: Finetune

With the following files, run `train.py`:
- `positive_proteins_<task_name>.json`
- `negative_proteins_<task_name>.json`
- `data_split_<task_name>.json`

Example command:
```
python train.py \
    --task_name EFO_0000685 \
    --embeddings_dir ../data/pinnacle_embeds/ \
    --embed pinnacle \
    --positive_proteins_prefix ../data/therapeutic_target_task/positive_proteins_EFO_0000685 \
    --negative_proteins_prefix ../data/therapeutic_target_task/negative_proteins_EFO_0000685 \
    --data_split_path ../data/therapeutic_target_task/data_split_EFO_0000685
```