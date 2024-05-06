# Main finetuning script

import pandas as pd
import numpy as np
import os, wandb, random

from setup import create_parser, get_hparams, setup_paths
from read_data import load_data
from train_utils import training_and_validation
from metrics_utils import get_metrics, save_torch_train_val_preds, save_results
from data_prep import process_and_split_data

import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import shuffle
os.environ['OPENBLAS_NUM_THREADS'] = '1'


def run_finetune(embed, celltype_dict, celltype_protein_dict, positive_proteins, negative_proteins, data_split_path, random_state, models_output_dir, embed_name, hparams, batch_size, num_epoch, train_size, val_size, weigh_sample, weigh_loss):

    # Training and validation
    X_train, X_test, y_train, y_test, groups_train, cts_train, groups_test = process_and_split_data(embed, positive_proteins, negative_proteins, celltype_protein_dict, celltype_dict, data_split_path, random_state=random_state, test_size=1-train_size-val_size)
    clf, best_epoch, best_val_auprc, train_ranks, val_ranks = finetune_train_stage(X_train, y_train, random_state, groups_train, cts_train, hparams, train_size, val_size, num_epoch, batch_size, weigh_sample, weigh_loss, models_output_dir, embed_name)
    
    positive_proportion_train = {}
    positive_proportion_train['celltype'] = sum(y_train) / len(y_train)
    wandb.log({f'train positive proportion celltype': positive_proportion_train['celltype'], 'best_val_auprc': best_val_auprc})

    # Evaluation for each cell celltype separately
    positive_proportion_test, auroc_scores, ap_scores = finetune_evaluate(celltype_protein_dict, clf, X_test, y_test, groups_test, models_output_dir, embed_name, train_ranks, val_ranks)

    # Save model
    save_path = os.path.join(models_output_dir, f"{embed_name}_model.pt")
    torch.save({'epoch': best_epoch, 'model_state_dict': clf.state_dict()}, save_path)

    return positive_proportion_train, positive_proportion_test, auroc_scores, ap_scores


def finetune_train_stage(X_train, y_train, random_state, groups_train, cts_train, hparams, train_size, val_size, num_epoch, batch_size, weigh_sample, weigh_loss, models_output_dir, embed_name):
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.from_numpy(X_train)

    n_splits = int((train_size+val_size)/val_size)
    train_indices, val_indices = list(StratifiedGroupKFold(n_splits=n_splits, random_state=random_state, shuffle=True).split(X=X_train, groups=groups_train, y=y_train))[np.random.randint(0, n_splits)]  # borrow CV generator to generate one split
    
    clf, best_train_y, best_train_preds, best_train_cts, best_train_groups, cts_map_train, groups_map_train, best_val_y, best_val_preds, best_val_cts, best_val_groups, cts_map_val, groups_map_val, best_epoch, best_val_auprc = training_and_validation(X_train[train_indices], X_train[val_indices], torch.Tensor(y_train)[train_indices], torch.Tensor(y_train)[val_indices], np.array(cts_train)[train_indices], np.array(cts_train)[val_indices], np.array(groups_train)[train_indices], np.array(groups_train)[val_indices], num_epoch, batch_size, weigh_sample, weigh_loss, hparams)
    train_ranks, val_ranks = save_torch_train_val_preds(best_train_y, best_train_preds, best_train_groups, best_train_cts, best_val_y, best_val_preds, best_val_groups, best_val_cts, groups_map_train, groups_map_val, cts_map_train, cts_map_val, models_output_dir, embed_name, wandb)

    clf = clf.to(torch.device('cpu'))
    return clf, best_epoch, best_val_auprc, train_ranks, val_ranks


def finetune_evaluate(celltype_protein_dict, clf, X_test, y_test, groups_test, models_output_dir, embed_name, train_ranks, val_ranks):
    auroc_scores = {}
    ap_scores = {}
    positive_proportion_test = {}

    test_ranks = {}
    for celltype in celltype_protein_dict:
        if celltype not in X_test: continue
        clf.eval()
        with torch.no_grad():
            y_test_pred = torch.sigmoid(clf(X_test[celltype])).squeeze(-1).numpy()
        
        # Evaluation on test set
        auroc_scores[celltype], ap_scores[celltype], test_recall_5, test_precision_5, test_ap_5, test_recall_10, test_precision_10, test_ap_10, sorted_y_test, sorted_preds_test, sorted_groups_test, positive_proportion_test[celltype] = get_metrics(y_test, y_test_pred, groups_test, celltype)
        wandb.log({f'test AUPRC {celltype}': ap_scores[celltype], 
                   f'test AUROC {celltype}': auroc_scores[celltype],
                   f'test positive proportion {celltype}': positive_proportion_test[celltype],
                   f'test positive number {celltype}': sum(y_test[celltype]),
                   f'test total number {celltype}': len(y_test[celltype]),
                   f'test recall@5 {celltype}': test_recall_5,
                   f'test precision@5 {celltype}': test_precision_5,
                   f'test AP@5 {celltype}': test_ap_5,
                   f'test recall@10 {celltype}': test_recall_10,
                   f'test precision@10 {celltype}': test_precision_10,
                   f'test AP@10 {celltype}': test_ap_10})
        temp = pd.DataFrame({'y': sorted_y_test, 'preds': sorted_preds_test, 'name': [prot for prot in sorted_groups_test]})
        temp['type'] = ['test'] * len(temp)
        test_ranks[celltype] = temp
        temp.to_csv(f'{models_output_dir}/{embed_name}_test_preds_{celltype}.csv', index=False)  # Save the test predictions
        combined_ranks = pd.concat([train_ranks[celltype], val_ranks[celltype], test_ranks[celltype]], axis=0).sort_values('preds', ascending=False)
        combined_ranks.to_csv(f'{models_output_dir}/{embed_name}_all_preds_{celltype}.csv', index=False)
    
    return positive_proportion_test, auroc_scores, ap_scores


def main(args, hparams, wandb):
    print(args)

    # Set up model environment and data/model paths
    models_output_dir, metrics_output_dir, random_state, embed_path, labels_path = setup_paths(args)
    
    # Load data
    embed, celltype_dict, celltype_protein_dict, positive_proteins, negative_proteins, _ = load_data(embed_path, labels_path, args.positive_proteins_prefix, args.negative_proteins_prefix, None, args.task_name)
    print("Finished reading data, evaluating...\n")

    # Run model
    data_split_path = args.data_split_path + ".json"
    positive_proportion_train, positive_proportion_test, auroc_scores, ap_scores = run_finetune(embed, celltype_dict, celltype_protein_dict, positive_proteins, negative_proteins, data_split_path, random_state, models_output_dir, args.embed, hparams, args.batch_size, args.num_epoch, args.train_size, args.val_size, args.weigh_sample, args.weigh_loss)

    # Generate outputs and plots
    print("Finished evaluation, generating plots...\n")
    output_results_path = os.path.join(metrics_output_dir, f"{args.embed}_{args.task_name}_results.csv")
    output_figs_path = os.path.join(metrics_output_dir, f"{args.embed}_{args.task_name}_")
    save_results(output_results_path, ap_scores, auroc_scores)
    

if __name__ == '__main__':
    args = create_parser()

    if not args.random:
        np.random.seed(args.random_state)
        random.seed(args.random_state)
        torch.manual_seed(args.random_state)
        torch.cuda.manual_seed(args.random_state)
        torch.backends.cudnn.deterministic = True

    hparams = get_hparams(args)
    print(hparams)

    wandb.init(config = hparams, project = "finetune", entity = "pinnacle")
    hparams = wandb.config

    main(args, hparams, wandb)