# Training and validation helper functions


import wandb
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import torch.nn as nn

from metrics_utils import get_metrics, precision_recall_at_k
from model import MLP
from copy import deepcopy


def training_and_validation(X_train, X_val, y_train, y_val, cts_train, cts_val, groups_train, groups_val, num_epoch, batch_size, weigh_sample, weigh_loss, hparams, no_val=False):
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
        _, _, train_y, train_preds, train_cts, train_groups = train_epoch(model, train_loader, optim, loss_func, batch_size, wandb, device)
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


def train_epoch(model, train_loader, optim, loss_func, batch_size, wandb, device):
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
            print(f"train loss: {loss:.4f} [{current}/{train_size}]")
            wandb.log({f"train loss":loss})

    print("Finished with batches...")

    all_y = all_y.detach().numpy().astype(int)
    all_preds = torch.sigmoid(all_preds).detach().numpy()
    all_cts = all_cts.detach().numpy().astype(int)
    all_groups = all_groups.detach().numpy().astype(int)

    train_auroc, train_auprc, train_recall_5, train_precision_5, train_ap_5, train_recall_10, train_precision_10, train_ap_10, _, _, _, _ = get_metrics(all_y, all_preds, all_groups, "training")

    total_loss = total_loss / total_sample
    wandb.log({f"train AUPRC": train_auprc,
               f"train AUROC": train_auroc,
               f"train recall@5": train_recall_5,
               f"train recall@10": train_recall_10,
               f"train precision@5": train_precision_5,
               f"train precision@10": train_precision_10,
               f"train AP@5": train_ap_5,
               f"train AP@10": train_ap_10})

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

    val_auroc, val_auprc, val_recall_5, val_precision_5, val_ap_5, val_recall_10, val_precision_10, val_ap_10, _, _, _, _ = get_metrics(ys, preds, groups, "training")

    wandb.log({f"val loss":val_loss,
               f"val AUPRC":val_auprc,
               f"val AUROC":val_auroc,
               f"val recall@5":val_recall_5,
               f"val recall@10":val_recall_10,
               f"val precision@5":val_precision_5,
               f"val precision@10":val_precision_10,
               f"val AP@5":val_ap_5,
               f"val AP@10":val_ap_10})

    print("Finished with calculating metrics...")
    return val_loss, val_auprc, ys, preds, cts, groups
