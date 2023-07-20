import pandas as pd
import numpy as np
import os, wandb, random

from train_utils import create_parser, get_hparams, loadData, setup, train_epoch, validate_epoch, training_and_validation
from metrics_utils import get_metrics, save_torch_train_val_preds, save_plots, save_results
from data_prep import process_and_split_data

import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import shuffle
os.environ['OPENBLAS_NUM_THREADS'] = '1'


def run_training(X_train, y_train, mod, random_state, groups_train, cts_train, hparams, train_size, val_size, num_epoch, batch_size, weigh_sample, weigh_loss, disease, models_output_dir, embed_name):
    """ 
    Main train function
    """
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.from_numpy(X_train)

    n_splits = int((train_size+val_size)/val_size)
    train_indices, val_indices = list(StratifiedGroupKFold(n_splits=n_splits, random_state=random_state, shuffle=True).split(X=X_train, groups=groups_train, y=y_train))[np.random.randint(0, n_splits)]  # borrow CV generator to generate one split
    
    clf, best_train_y, best_train_preds, best_train_cts, best_train_groups, cts_map_train, groups_map_train, best_val_y, best_val_preds, best_val_cts, best_val_groups, cts_map_val, groups_map_val, best_epoch, best_val_auprc = training_and_validation(X_train[train_indices], X_train[val_indices], torch.Tensor(y_train)[train_indices], torch.Tensor(y_train)[val_indices], np.array(cts_train)[train_indices], np.array(cts_train)[val_indices], np.array(groups_train)[train_indices], np.array(groups_train)[val_indices], disease, num_epoch, batch_size, weigh_sample, weigh_loss, hparams)
    train_ranks, val_ranks = save_torch_train_val_preds(best_train_y, best_train_preds, best_train_groups, best_train_cts, best_val_y, best_val_preds, best_val_groups, best_val_cts, groups_map_train, groups_map_val, cts_map_train, cts_map_val, models_output_dir, embed_name, disease, mod, wandb)

    clf = clf.to(torch.device('cpu'))
    return clf, best_epoch, best_val_auprc, train_ranks, val_ranks


def evaluate_disease_gene_association(embed, subtype_dict, subtype_protein_dict, disease_positive_proteins, disease_negative_proteins, data_split_path, mod, random_state, therapeutic_area, models_output_dir, embed_name, hparams, batch_size, num_epoch, train_size, val_size, weigh_sample, weigh_loss, celltype_only):
    """
    Evaluate disease-drug interaction using cell type specific embeddings
    """
    positive_proportion_test = {}
    positive_proportion_train = {}
    ap_scores = {}
    auroc_scores = {}
    mod_hparams = {}
    celltype_subtype = list(subtype_protein_dict.keys())
    
    for disease in [therapeutic_area]:
        print(f"Working on disease: {disease}...\n")
        
        # Training and validation
        X_train, X_test, y_train, y_test, groups_train, cts_train, groups_test = process_and_split_data(embed, disease_positive_proteins, disease_negative_proteins, subtype_protein_dict, celltype_subtype, subtype_dict, disease, data_split_path, random_state=random_state, test_size=1-train_size-val_size)
        clf, best_epoch, best_val_auprc, train_ranks, val_ranks = run_training(X_train, y_train, mod, random_state, groups_train, cts_train, hparams, train_size, val_size, num_epoch, batch_size, weigh_sample, weigh_loss, False, disease, models_output_dir, embed_name)

        # Evaluation for each cell celltype separately
        auroc_scores[disease] = {}
        ap_scores[disease] = {}
        mod_hparams[disease] = {}
        positive_proportion_test[disease] = {}
        positive_proportion_train[disease] = {}
        positive_proportion_train[disease]['celltype'] = sum(y_train) / len(y_train)
        wandb.log({f'train positive proportion celltype': positive_proportion_train[disease]['celltype'], 'best_val_auprc': best_val_auprc})

        # Testing
        test_ranks = {}
        for celltype in celltype_subtype:
            if celltype not in X_test: continue
            clf.eval()
            with torch.no_grad():
                y_test_pred = torch.sigmoid(clf(X_test[celltype])).squeeze(-1).numpy()
            
            # Evaluation on test set
            auroc_scores[disease][celltype], ap_scores[disease][celltype], test_recall_5, test_precision_5, test_ap_5, test_recall_10, test_precision_10, test_ap_10, test_recall_20, test_precision_20, test_ap_20, sorted_y_test, sorted_preds_test, sorted_groups_test, positive_proportion_test[disease][celltype] = get_metrics(y_test, y_test_pred, groups_test, celltype)
            wandb.log({f'test AUPRC {celltype}': ap_scores[disease][celltype], 
                       f'test AUROC {celltype}': auroc_scores[disease][celltype],
                       f'test positive proportion {celltype}': positive_proportion_test[disease][celltype],
                       f'test positive number {celltype}': sum(y_test[celltype]),
                       f'test total number {celltype}': len(y_test[celltype]),
                       f'test recall@5 {celltype}': test_recall_5,
                       f'test precision@5 {celltype}': test_precision_5,
                       f'test AP@5 {celltype}': test_ap_5,
                       f'test recall@10 {celltype}': test_recall_10,
                       f'test precision@10 {celltype}': test_precision_10,
                       f'test AP@10 {celltype}': test_ap_10,
                       f'test recall@20 {celltype}': test_recall_20,
                       f'test precision@20 {celltype}': test_precision_20,
                       f'test AP@20 {celltype}': test_ap_20})
            temp = pd.DataFrame({'y': sorted_y_test, 'preds': sorted_preds_test, 'name': [prot for prot in sorted_groups_test]})
            temp['type'] = ['test'] * len(temp)
            test_ranks[celltype] = temp
            temp.to_csv(f'{models_output_dir}/{embed_name}_{disease}_{mod}_test_preds_{celltype}.csv', index=False)  # Save the test predictions
            combined_ranks = pd.concat([train_ranks[celltype], val_ranks[celltype], test_ranks[celltype]], axis=0).sort_values('preds', ascending=False)
            combined_ranks.to_csv(f'{models_output_dir}/{embed_name}_{disease}_{mod}_all_preds_{celltype}.csv', index=False)

        # Save model
        save_path = os.path.join(models_output_dir, f"{embed_name}_{disease}_{mod}_model.pt")
        torch.save({'epoch': best_epoch, 'model_state_dict': clf.state_dict()}, save_path)

    return positive_proportion_train, positive_proportion_test, auroc_scores, ap_scores


def main(args, hparams, wandb):
    print(args)

    # Set up model environment and data/model paths
    models_output_dir, metrics_output_dir, random_state, embed_path, labels_path = setup(args)
    
    # Load data
    embed, subtype_dict, subtype_protein_dict, positive_proteins, negative_proteins, _ = loadData(embed_path, labels_path, args.disease, hparams["celltypes"], overwrite=args.overwrite, wandb=wandb, args=args)
    print("Finished reading data, evaluating...\n")

    # Run model
    print(f"Running model: {args.model}")
    if len(hparams["celltypes"]) == 0:
        data_split_path = args.data_split_path + args.disease + ".json"
    else:
        data_split_path = args.data_split_path + args.disease + "_" + args.celltypes + ".json"
    positive_proportion_train, positive_proportion_test, auroc_scores, ap_scores = evaluate_disease_gene_association(embed, subtype_dict, subtype_protein_dict, positive_proteins, negative_proteins, data_split_path, args.model, random_state, args.disease, models_output_dir, args.embed, hparams, args.batch_size, args.num_epoch, args.train_size, args.val_size, args.weigh_sample, args.weigh_loss, hparams["celltype_only"])

    # Generate outputs and plots
    print("Finished evaluation, generating plots...\n")
    output_results_path = os.path.join(metrics_output_dir, f"{args.embed}_{args.disease}_{args.model}_results.csv")
    output_figs_path = os.path.join(metrics_output_dir, f"{args.embed}_{args.disease}_{args.model}_")
    save_plots(output_figs_path, positive_proportion_train, positive_proportion_test, ap_scores, auroc_scores, args.disease, wandb)
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

    wandb.init(config = hparams, project = "tx-target", entity = "michellemli")
    hparams = wandb.config

    main(args, hparams, wandb)