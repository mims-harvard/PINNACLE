# Evaluating PINNACLE

## Visualize embeddings

```
python visualize_representations.py 
```


## Therapeutic target task

### Summary plot

Generate summary plot across cell types for a specific therapeutic area of interest

#### Rheumatoid arthritis (EFO_0000685)
```
python evaluate_target_prioritization.py --disease EFO_0000685 --model_outputs_dir ../pinnacle/model_outputs/ --test_only True
```

#### Inflammatory bowel diseases (EFO_0003767)
```
python evaluate_target_prioritization.py --disease EFO_0003767 --model_outputs_dir ../pinnacle/model_outputs/ --test_only True
```

### Benchmarking

To include benchmarks into the plot, you need a benchmark inventory, which is a list of csv files output by the model (csv path per line in the inventory document). The expected format of these csv files is the same as that of finetuned PINNACLE.

An example command-line to extract the appropriate files of the benchmark models is:
```
ls "$PWD"/TS_seed*/benchmarks/*all_pred*csv > benchmark_inventory.txt
```

Finally, you can use the `--benchmark_inventory` flag to provide the benchmark inventory file.

#### Rheumatoid arthritis (EFO_0000685)
```
python evaluate_target_prioritization.py --disease EFO_0000685 --model_outputs_dir ../pinnacle/model_outputs/ --test_only True --benchmark_inventory ../data/therapeutic_target_task/benchmark_inventory.txt
```

#### Inflammatory bowel diseases (EFO_0003767)
```
python evaluate_target_prioritization.py --disease EFO_0003767 --model_outputs_dir ../pinnacle/model_outputs/ --test_only True --benchmark_inventory ../data/therapeutic_target_task/benchmark_inventory.txt
```

### Generate per-target plot

Generate a plot of the top and bottom 5 cell types to evaluate a given protein target

#### Rheumatoid arthritis (EFO_0000685): JAK3, IL6R
```
python evaluate_target_prioritization.py --disease EFO_0000685 --model_outputs_dir ../pinnacle/model_outputs/ --drug_targets JAK3,IL6R --seeds 3
```

#### Inflammatory bowel diseases (EFO_0003767): ITGA4, PPARG
```
python evaluate_target_prioritization.py --disease EFO_0003767 --model_outputs_dir ../pinnacle/model_outputs/ --drug_targets ITGA4,PPARG --seeds 5
```