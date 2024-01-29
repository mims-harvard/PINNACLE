import torch
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calc_link_pred_loss(mg_pred, mg_y, ppi_preds, ppi_y, loss_type="BCE"):
    
    # Calculate link prediction loss on metagraph
    mg_loss = 0
    if len(mg_pred) > 0:
        mg_loss = F.binary_cross_entropy(mg_pred, mg_y["y"].to(device))

    # Calculate link prediction loss on PPI networks
    ppi_loss = 0
    for celltype, ppi in ppi_preds.items():
        ppi_loss += F.binary_cross_entropy(ppi, ppi_y[celltype]["y"].to(device))

    return ppi_loss, mg_loss


def calc_center_loss(center_loss, embed, centers, y, mask):
    loss = center_loss(embed[mask, :], centers, y[mask].to(device))
    return loss


def max_margin_loss(pred, y):
    loss = (1 - (pred[y == 1] - pred[y != 1])).clamp(min=0)
    return loss


def el_dot(embed, edges, relation): 
    source = embed[edges[0, :]]
    target = embed[edges[1, :]]
    if len(relation) != 0: dots = torch.sum(source * relation * target, dim = 1)
    else: dots = torch.sum(source * target, dim = 1)
    return torch.sigmoid(dots)
