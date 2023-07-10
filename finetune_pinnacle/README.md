# Finetuning PINNACLE for downstream tasks

## Step-by-step process on finetuning PINNACLE to nominate therapeutic targets for a specified therapeutic area

### :one: Prepare dataset for a specified therapeutic area

```
python data_prep.py --celltype_ppi PATH/TO/PPI --disease DISEASE_NAME --disease_drug_evidence_prefix targets/disease_drug_evidence_ --positive_proteins_prefix targets/positive_proteins_ --negative_proteins_prefix targets/negative_proteins_ --raw_targets_prefix targets/raw_targets_ --all_drug_targets_path targets/all_approved.csv
```

### :two: Train model on a specified therapeutic area

```
python train.py --actn=relu --disease=EFO_0000685 --dropout=0.2 --embeddings_dir=/PATH/TO/PINNACLE/EMB --hidden_dim_1=32 --hidden_dim_2=8 --lr=0.01 --norm=bn --order=dn --wd=0.001 --random_state 1 --num_epoch=2000
```
