############################################################
#
# Set up model environment, parameters, and data
#
############################################################


import os
import pandas as pd
import argparse
import json
import pickle
import torch

from data_prep import get_labels_from_evidence


def create_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, default="torch_mlp")

    parser.add_argument("--hidden_dim_1", type=int, default=64, help="1st hidden dim size")
    parser.add_argument("--hidden_dim_2", type=int, default=32, help="2nd hidden dim size, discard if 0")
    parser.add_argument("--hidden_dim_3", type=int, default=0, help="3rd hidden dim size, discard if 0")
    parser.add_argument("--dropout", type=float, default=0, help="dropout rate")
    parser.add_argument("--norm", type=str, default=None, help="normalization layer")
    parser.add_argument("--actn", type=str, default="relu", help="activation type")
    parser.add_argument("--order", type=str, default="nd", help="order of normalization and dropout")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--num_epoch", type=int, default=1000, help="epoch num")
    parser.add_argument("--batch_size", type=int, help="batch size")

    # Input therapeutic targets data
    parser.add_argument("--curated_disease_dir", type=str, default="/n/data1/hms/dbmi/zitnik/lab/datasets/2022-06-Open_Targets/data/diseases/")
    parser.add_argument("--evidence_dir", type=str, default="/n/data1/hms/dbmi/zitnik/lab/datasets/2022-06-Open_Targets/data/evidence/sourceId=chembl/")
    parser.add_argument("--chembl2db_path", type=str, default="/n/data1/hms/dbmi/zitnik/lab/datasets/2022-06-Open_Targets/data/chembl2db.txt")  # Download mapping from ChEMBL id to DrugBank id from https://ftp.ebi.ac.uk/pub/databases/chembl/UniChem/data/wholeSourceMapping/src_id1/src1src2.txt (version: 13-Apr-2022)
    parser.add_argument("--positive_proteins_prefix", type=str, default="targets/positive_proteins_") # Output from data_prep.py
    parser.add_argument("--negative_proteins_prefix", type=str, default="targets/negative_proteins_") # Output from data_prep.py
    parser.add_argument("--raw_targets_prefix", type=str, default="targets/raw_targets_") # Output from data_prep.py
    parser.add_argument("--all_drug_targets_path", type=str, default="targets/all_approved.csv") # Output from data_prep.py
    parser.add_argument("--data_split_path", type=str, default="targets/data_split_")

    # Output directories
    parser.add_argument("--metrics_output_dir", type=str, default="./tmp_evaluation_results/")
    parser.add_argument("--models_output_dir", type=str, default="./tmp_model_outputs/")
    
    parser.add_argument("--embeddings_dir", type=str)
    parser.add_argument("--embed", type=str, default="pinnacle")
    parser.add_argument("--globe", type=str, default="")
    parser.add_argument("--esm", type=str, default="")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--random", action="store_true", help="random runs without fixed seeds")
    parser.add_argument("--overwrite", action="store_true", help="whether to overwrite the label data or not")
    parser.add_argument("--train_size", type=float, default=0.6)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--weigh_sample", action="store_true", help="whether to weigh samples or not")  # default = False
    parser.add_argument("--weigh_loss", action="store_true", help="whether to weigh losses or not")  # default = False
    
    parser.add_argument("--disease", type=str)
    parser.add_argument("--celltypes", type=str, default="")
    args = parser.parse_args()
    return args


def get_hparams(args):

    def parse_celltype_arg():
        if args.celltypes == "": return []
        celltypes = []
        for c in args.celltypes.split(","):
            celltypes.append(" ".join(c.split("_")))
        return celltypes


    hparams = {
               "lr": args.lr, 
               "wd": args.wd, 
               "hidden_dim_1": args.hidden_dim_1, 
               "hidden_dim_2": args.hidden_dim_2, 
               "hidden_dim_3": args.hidden_dim_3, 
               "dropout": args.dropout, 
               "actn": args.actn, 
               "order": args.order, 
               "norm": args.norm,
               "celltype_only": True if args.globe == "" else False,
               "disease": args.disease,
               "celltypes": parse_celltype_arg(),
              }
    return hparams


def setup(args):
    random_state = args.random_state if args.random_state >= 0 else None
    if random_state == None:
        models_output_dir = args.models_output_dir + args.embed + "/"
        metrics_output_dir = args.metrics_output_dir + args.embed + "/"
    else:
        models_output_dir = args.models_output_dir + args.embed + ("_seed=%s" % str(random_state)) + "/"
        metrics_output_dir = args.metrics_output_dir + args.embed + ("_seed=%s" % str(random_state)) + "/"
    if not os.path.exists(models_output_dir): os.makedirs(models_output_dir)
    if not os.path.exists(metrics_output_dir): os.makedirs(metrics_output_dir)
    
    embed_path = args.embeddings_dir + args.embed + "_ppi_embed.pth"
    labels_path = args.embeddings_dir + args.embed + "_labels_dict.txt"
    return models_output_dir, metrics_output_dir, random_state, embed_path, labels_path


def loadData(embed_path: str, labels_path: str, therapeutic_area: str, celltype_list: list, overwrite: bool, wandb, args):
    
    embed = torch.load(embed_path)
    with open(labels_path, "r") as f:
        labels_dict = f.read()
    labels_dict = labels_dict.replace("\'", "\"")
    labels_dict = json.loads(labels_dict)
    subtypes = [c for c in labels_dict["Cell Type"] if c.startswith("CCI")]
    subtype_dict = {ct.split("CCI_")[1]: i for i, ct in enumerate(subtypes)}
    assert len(subtype_dict) > 0
    
    protein_names = []
    protein_celltypes = []
    for c, p in zip(labels_dict["Cell Type"], labels_dict["Name"]):
        if c.startswith("BTO") or c.startswith("CCI") or c.startswith("Sanity"): continue
        protein_names.append(p)
        protein_celltypes.append(c)

    proteins = pd.DataFrame.from_dict({"target": protein_names, "cell type": protein_celltypes})
    subtype_protein_dict = proteins.pivot_table(values="target", index="cell type", aggfunc={"target": list}).to_dict()["target"]
    assert len(subtype_protein_dict) > 0

    disease_positive_proteins, disease_negative_proteins, clinically_relevant_targets = get_labels_from_evidence(subtype_protein_dict, [therapeutic_area], args.evidence_dir, args.all_drug_targets_path, args.curated_disease_dir, args.chembl2db_path, args.positive_proteins_prefix, args.negative_proteins_prefix, args.raw_targets_prefix, overwrite, "", wandb)

    print("Positive proteins collected for", disease_positive_proteins.keys())
    print("Negative proteins collected for", disease_negative_proteins.keys())

    print("Celltypes to filter for", celltype_list)

    if len(celltype_list) > 0:
        filtered_disease_positive_proteins = {therapeutic_area: dict()}
        filtered_disease_negative_proteins = {therapeutic_area: dict()}
        assert len(filtered_disease_positive_proteins[therapeutic_area].keys()) == len(celltype_list) + 1
        assert len(filtered_disease_negative_proteins[therapeutic_area].keys()) == len(celltype_list) + 1
        disease_positive_proteins = filtered_disease_positive_proteins
        disease_negative_proteins = filtered_disease_negative_proteins
        subtype_protein_dict = {k: v for k, v in subtype_protein_dict.items() if k in disease_positive_proteins[therapeutic_area]}
    assert len(disease_positive_proteins[therapeutic_area]) > 0

    return embed, subtype_dict, subtype_protein_dict, disease_positive_proteins, disease_negative_proteins, clinically_relevant_targets







############################################################
#
# Training and validation helper functions
#
############################################################


import wandb
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import torch.nn as nn

from metrics_utils import get_metrics, precision_recall_at_k
from model import MLP
from copy import deepcopy


def training_and_validation(X_train, X_val, y_train, y_val, cts_train, cts_val, groups_train, groups_val, disease, num_epoch, batch_size, weigh_sample, weigh_loss, hparams, no_val=False):
    """
    Train an MLP on a train-val split.
    """
    norm = "norm"
    actn = "actn"
    hidden_dim_1 = "hidden_dim_1"
    hidden_dim_2 = "hidden_dim_2"
    hidden_dim_3 = "hidden_dim_3"
    dropout = "dropout"
    lr = "lr"
    wd = "wd"
    order = "order"
    best_val_auprc = 0

    if not no_val:
        cts_map_val = np.unique(cts_val, return_inverse=True)[0]  # factorize cts_val
        cts_val = np.unique(cts_val, return_inverse=True)[1]
        groups_map_val = np.unique(groups_val, return_inverse=True)[0]  # factorize groups_val
        groups_val = np.unique(groups_val, return_inverse=True)[1]
    
    cts_map_train = np.unique(cts_train, return_inverse=True)[0]  # factorize cts_train
    cts_train = np.unique(cts_train, return_inverse=True)[1]
    groups_map_train = np.unique(groups_train, return_inverse=True)[0]  # factorize groups_train
    groups_train = np.unique(groups_train, return_inverse=True)[1]

    train_dataset = TensorDataset(X_train, y_train.unsqueeze(-1), torch.from_numpy(cts_train), torch.from_numpy(groups_train))
    if not no_val:
        val_dataset = TensorDataset(X_val, y_val.unsqueeze(-1), torch.from_numpy(cts_val), torch.from_numpy(groups_val))

    sampler = None
    shuffle = True
    if weigh_sample:
        class_sample_num = torch.unique(y_train, return_counts=True)[1]
        weights = torch.DoubleTensor([1/class_sample_num[y.int().item()] for y in y_train])
        sampler = WeightedRandomSampler(weights, len(weights))
        shuffle = False
    drop_last = False
    if batch_size is None:
        batch_size = len(train_dataset)
    if (hparams[norm] == "bn" or hparams[norm] == "ln") and len(train_dataset) % batch_size < 3:
        drop_last = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle, num_workers=2, drop_last=drop_last)
    if not no_val:
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=2)  # set val batch size to full-batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hparams[hidden_dim_2] == 0:
        hidden_dims = [hparams[hidden_dim_1]]
    elif hparams[hidden_dim_3] == 0:
        hidden_dims = [hparams[hidden_dim_1], hparams[hidden_dim_2]]
    else:
        hidden_dims = [hparams[hidden_dim_1], hparams[hidden_dim_2], hparams[hidden_dim_3]]

    model = MLP(in_dim = X_train.shape[1], hidden_dims = hidden_dims, p = hparams[dropout], norm=hparams[norm], actn=hparams[actn], order=hparams[order])
    model = model.to(device)

    pos_weight = None
    if weigh_loss:
        pos_weight = torch.Tensor([(y_train.shape[0] - y_train.sum().item()) / y_train.sum().item()]).to(device)
    loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=hparams[lr], weight_decay=hparams[wd])
    
    wandb.watch(model, log_freq=20)
    for i in range(num_epoch):
        print(f"Epoch {i+1}\n---------------")
        _, _, train_y, train_preds, train_cts, train_groups = train_epoch(model, train_loader, optim, loss_func, batch_size, wandb, device, disease)
        if not no_val:
            _, val_auprc, val_y, val_preds, val_cts, val_groups = validate_epoch(model, val_loader, loss_func, wandb, device)
            if val_auprc > best_val_auprc:
                clf = deepcopy(model)
                best_val_auprc = val_auprc
                best_epoch = i
                best_val_groups = val_groups.copy().astype(int)
                best_val_y = val_y.copy()
                best_val_preds = val_preds.copy()
                best_train_groups = train_groups.copy().astype(int)
                best_train_y = train_y.copy()
                best_train_preds = train_preds.copy()
                best_val_cts = val_cts.copy().astype(int)
                best_train_cts = train_cts.copy().astype(int)

    if no_val:
        best_train_groups = train_groups.astype(int)
        best_train_y = train_y
        best_train_preds = train_preds
        best_train_cts = train_cts.astype(int)
        return model, best_train_y, best_train_preds, best_train_cts, best_train_groups, cts_map_train, groups_map_train
    
    return clf, best_train_y, best_train_preds, best_train_cts, best_train_groups, cts_map_train, groups_map_train, best_val_y, best_val_preds, best_val_cts, best_val_groups, cts_map_val, groups_map_val, best_epoch, best_val_auprc


def train_epoch(model, train_loader, optim, loss_func, batch_size, wandb, device, disease):
    model.train()
    train_size = len(train_loader.dataset)
    total_sample = total_loss = 0
    all_y = torch.tensor([])
    all_preds = torch.tensor([])
    all_cts = torch.tensor([])
    all_groups = torch.tensor([])
    for i, (X, y, cts, groups) in enumerate(train_loader):
        print("Batch", i)

        all_y = torch.cat([all_y, y])
        all_cts = torch.cat([all_cts, cts])
        all_groups = torch.cat([all_groups, groups])

        X, y = X.to(device), y.to(device)
        optim.zero_grad()
        preds = model(X)
        loss = loss_func(preds, y)
        
        loss.backward()
        optim.step()

        all_preds = torch.cat([all_preds, preds.cpu()])
        total_sample += batch_size
        total_loss += float(loss) * batch_size

        if i % 20 == 0:
            loss, current = loss.item(), i * len(X)
            print(f"train loss {disease}: {loss:.4f} [{current}/{train_size}]")
            wandb.log({f"train loss":loss})

    print("Finished with batches...")

    all_y = all_y.detach().numpy().astype(int)
    all_preds = torch.sigmoid(all_preds).detach().numpy()  # Need to do a sigmoid here
    all_cts = all_cts.detach().numpy().astype(int)
    all_groups = all_groups.detach().numpy().astype(int)

    train_auroc, train_auprc, train_recall_5, train_precision_5, train_ap_5, train_recall_10, train_precision_10, train_ap_10, train_recall_20, train_precision_20, train_ap_20, _, _, _, _ = get_metrics(all_y, all_preds, all_groups, "training")
    train_recall_50, train_precision_50, train_ap_50, _ = precision_recall_at_k(all_y, all_preds, k=50)

    total_loss = total_loss / total_sample  # weighted total train loss
    wandb.log({f"train AUPRC": train_auprc,
               f"train AUROC": train_auroc,
               f"train recall@5": train_recall_5,
               f"train recall@10": train_recall_10,
               f"train recall@20": train_recall_20,
               f"train recall@50": train_recall_50,
               f"train precision@5": train_precision_5,
               f"train precision@10": train_precision_10,
               f"train precision@20": train_precision_20,
               f"train precision@50": train_precision_50,
               f"train AP@5": train_ap_5,
               f"train AP@10": train_ap_10,
               f"train AP@20": train_ap_20,
               f"train AP@50": train_ap_50})

    print("Finished with one full epoch...")

    return total_loss, train_auprc, all_y, all_preds, all_cts, all_groups


@torch.no_grad()
def validate_epoch(model, val_loader, loss_func, wandb, device):
    val_size = len(val_loader.dataset)

    model.eval()
    val_loss = 0
    all_y = torch.tensor([])
    all_preds = torch.tensor([])
    all_cts = torch.tensor([])
    all_groups = torch.tensor([])
    
    for X, y, cts, groups in val_loader:
        all_y = torch.cat([all_y, y])
        X, y = X.to(device), y.to(device)
        all_cts = torch.cat([all_cts, cts])
        all_groups = torch.cat([all_groups, groups])

        preds = model(X)
        all_preds = torch.cat([all_preds, preds.cpu()])
        val_loss += loss_func(preds, y).item() * X.shape[0]

    print("Finished all batches in validation...")
    
    val_loss /= val_size

    ys, preds, cts, groups = all_y.detach().numpy(), torch.sigmoid(all_preds).detach().numpy(), all_cts.detach().numpy(), all_groups.detach().numpy()

    val_auroc, val_auprc, val_recall_5, val_precision_5, val_ap_5, val_recall_10, val_precision_10, val_ap_10, val_recall_20, val_precision_20, val_ap_20, _, _, _, _ = get_metrics(ys, preds, groups, "training")
    val_recall_50, val_precision_50, val_ap_50, _ = precision_recall_at_k(ys, preds, k=50)

    wandb.log({f"val loss":val_loss, f"val AUPRC":val_auprc, f"val AUROC":val_auroc, f"val recall@5":val_recall_5, f"val recall@10":val_recall_10, f"val recall@20":val_recall_20, f"val recall@50":val_recall_50, f"val precision@5":val_precision_5, f"val precision@10":val_precision_10, f"val precision@20":val_precision_20, f"val precision@50":val_precision_50, f"val AP@5":val_ap_5, f"val AP@10":val_ap_10, f"val AP@20":val_ap_20, f"val AP@50":val_ap_50})

    print("Finished with calculating metrics...")
    return val_loss, val_auprc, ys, preds, cts, groups
