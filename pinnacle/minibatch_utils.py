import random
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, GraphSAINTRandomWalkSampler, GraphSAINTEdgeSampler
from torch_geometric.utils import structured_negative_sampling

from utils import construct_metapath, get_embeddings
from loss import el_dot, calc_link_pred_loss, calc_center_loss


def pred_batch2dict(packed_batch: object, mg_x_ori: dict, ppi_x_ori: dict, cell_type_order: list, device: str) -> dict:
    """
    Re-initialize :code:`ppi_x`, :code:`metagraph`, and transform packed batches of all graphs to a dictionary of batches. Note that different from :code:`train_batch2dict`, we are also re-initializing full :code:`ppi_x` because we are feeding all nodes instead of the sampled nodes in the batch to the model during prediction.
    
    :param packed_batch: An iterable (tuple if directly following unpacking of the output of :code:`generatePPIBatch`) storing batches of :class:`Data` from all graphs in one round.
    :param mg_x_ori: metagraph original node embeddings.
    :param ppi_x_ori: Original all PPI node embeddings
    :param cell_type_order: Cell type order.
    :param device: A string indicating the device. Default is "cuda".
    
    :return: A dictionary of edge data storing batches from all graphs in one round, :code:`ppi_x_batch`, :code:`ppi_node_ind_batch` extracted from batches, and the re-initialized node embeddings :code:`mg_x_init`.
    """
    # Re-initalize mg_x from mg_x in each batch
    ppi_x_init = {key: x.clone().to(device) for key, x in ppi_x_ori.items()}
    mg_x_init = mg_x_ori.clone().to(device) if len(mg_x_ori)!=0 else []
    
    # Unpack batches
    ppi_data_batch = {key: {} for key in cell_type_order}
    for ind, batch in enumerate(packed_batch):
        i = cell_type_order[ind]
        ori_map = batch.n_id
        ppi_data_batch[i]['total_edge_index'] = ori_map[batch.edge_index].to(device)
        ppi_data_batch[i]['total_edge_type'] = batch.edge_attr.to(device)
        ppi_data_batch[i]['y'] = batch.y.to(device)
    
    return ppi_data_batch, ppi_x_init, mg_x_init


def iterate_train_batch(ppi_train_loader_dict: dict, ppi_x_ori: dict, ppi_metapaths_ori: dict, mg_x_ori: dict,  mg_metapaths_train: list, mg_data_train: dict, tissue_neighbors: dict, model: torch.nn.Module, hparams: dict, device: str, wandb: object=None, center_loss: torch.nn.Module=None, optimizer: torch.optim=None, mask_train_ori: list=None) -> tuple:
    """
    Iterate batches for train. In each batch, only embeddings of nodes corresponding to the sampled edges (i.e., sampled nodes and their 2-hop neighbors) are attention-pooled to approximate the global embedding of a cell type's PPI, and used to update the node embedding in CCI. 
    
    :return: :code:`ppi_x_out`, :code:`mg_x`, :code:`mg_pred`, :code:`ppi_preds_all`, :code:`ppi_data_y`, and :code:`total_loss`.
    """
    total_samples = total_loss = 0
    ppi_preds_all = {}
    ppi_data_y = {key:{'y': torch.tensor([]), 'total_edge_type': torch.tensor([])} for key in ppi_x_ori.keys()}
    ppi_x_out = {key: torch.zeros((x.shape[0], model.output)) for key, x in ppi_x_ori.items()}
    count = 0

    # START BATCH FOR LOOP
    for packed_batch in zip(*ppi_train_loader_dict.values()):
        count += 1
        
        print(f"Training batch {count}")
        optimizer.zero_grad()
        
        # Unpack batches to edges, nodes, and indices, and reinitialize mg_x
        ppi_data_batch, ppi_x, ppi_node_ind_batch, ppi_metapaths_batch, _ = train_batch2dict(packed_batch, mg_x_ori, ppi_metapaths_ori, list(ppi_train_loader_dict.keys()), device)
        batch_size = sum([data['y'].shape[0] for data in ppi_data_batch.values()])  # Number of all samples across all cell types
        
        # Generate PPI and metagraph embeddings & Compute predictions for metagraph
        ppi_x, mg_x = model(ppi_x, mg_x_ori, ppi_metapaths_batch, mg_metapaths_train, ppi_data_batch, mg_data_train["total_edge_index"], tissue_neighbors)

        # Compute predictions for metagraph for train
        mg_pred = el_dot(mg_x, mg_data_train["total_edge_index"], model.mg_relw[mg_data_train["total_edge_type"]])
        
        # Compute predictions for PPI layers
        ppi_preds = dict()
        for celltype, x in ppi_x.items():
            ppi_preds[celltype] = el_dot(x, ppi_data_batch[celltype]['total_edge_index'], [])
            ppi_preds_all[celltype] = torch.cat([ppi_preds_all.setdefault(celltype, torch.tensor([])), ppi_preds[celltype].detach().cpu()])
            ppi_data_y[celltype]['y'] = torch.cat([ppi_data_y[celltype]['y'], ppi_data_batch[celltype]['y'].detach().cpu()])
            ppi_data_y[celltype]['total_edge_type'] = torch.cat([ppi_data_y[celltype]['total_edge_type'], ppi_data_batch[celltype]['total_edge_type'].detach().cpu()])
            ppi_x_out[celltype][ppi_node_ind_batch[celltype]] = x.detach().cpu()

        # Compute train loss
        ppi_loss, mg_loss = calc_link_pred_loss(mg_pred, mg_data_train, ppi_preds, ppi_data_batch, hparams['loss_type'])
        link_loss = hparams['theta'] * ppi_loss + (1 - hparams['theta']) * mg_loss

        # Get embeddings
        embed = torch.cat(list(ppi_x.values())) # Protein
        centers = mg_x[0:len(ppi_x)] # Cell type

        # Protein labels
        center_loss_labels = torch.cat([(torch.ones(x.shape[0]) * key).to(torch.long) for key, x in ppi_x.items()])  # Build center loss labels based on batched nodes to ensure consistency with embedding labels

        # Train mask
        train_mask = construct_batch_center_loss_mask(mask_train_ori, ppi_node_ind_batch, ppi_x_ori)
        
        # Center loss
        cent_loss = calc_center_loss(center_loss, embed, centers, center_loss_labels, train_mask)
        print("Link Prediction: ", link_loss, "Center Loss: ", cent_loss)
        wandb.log({"Link Prediction Loss": link_loss, "Center Loss": cent_loss})
        combined_loss = link_loss + (cent_loss * hparams["lambda"])
        combined_loss.backward()
        
        # Update
        for param in center_loss.parameters():
            param.grad.data *= (hparams["lr_cent"] / (hparams["lambda"] * hparams["lr"]))
        if hparams['gradclip'] != -1: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), hparams['gradclip'])
        optimizer.step()
        
        # Calculate loss
        total_samples += batch_size
        total_loss += float(combined_loss) * batch_size
        # Note that here for simplicity the total loss rather than only the link prediction BCEloss is weighted by edge batch size. 

    total_loss = total_loss/total_samples  # Weighted total train loss
    
    return ppi_x_out, mg_x, mg_pred, ppi_preds_all, ppi_data_y, total_loss
    

def iterate_predict_batch(ppi_loader_dict: dict, ppi_x_ori: dict, ppi_metapaths_eval: dict, mg_x_ori: dict,  mg_metapaths: list, mg_data: dict, tissue_neighbors: dict, model: torch.nn.Module, hparams: dict, device: str) -> tuple:
    """
    Iterate batches for prediction (val/test). To ensure consistent results, the full :code:`ppi_x` is being updated each round with train (for validation), or train & val metapaths (for test), respectively. Minibatching is only performed for edges used for link prediction here to reduce memory cost. Setting val/test batch num to 1 is recommended wherever probable.
    
    :return: :code:`ppi_x`, :code:`mg_x`, :code:`mg_pred`, :code:`ppi_preds_all`, and :code:`ppi_data_y`.
    """
    ppi_preds_all = {}
    ppi_data_y = {key:{'y':torch.tensor([]), 'total_edge_type':torch.tensor([])} for key in ppi_x_ori.keys()}
    count = 0
    
    for packed_batch in zip(*ppi_loader_dict.values()):
        count += 1
        
        # Unpack batches and reinitialize mg_x
        ppi_data_batch, ppi_x, mg_x = pred_batch2dict(packed_batch, mg_x_ori, ppi_x_ori, list(ppi_loader_dict.keys()), device)

        # Generate PPI and metagraph embeddings & Compute predictions for metagraph
        if mg_data["total_edge_index"] !=  []: mg_data["total_edge_index"] = mg_data["total_edge_index"].to(device)
        ppi_x, mg_x = get_embeddings(model.to(device), ppi_x, mg_x, ppi_metapaths_eval, mg_metapaths, ppi_data_batch, mg_data["total_edge_index"], tissue_neighbors)

        # Compute predictions for metagraph for val/test only once
        if count == 1:
            mg_pred = el_dot(mg_x.to(device), mg_data["total_edge_index"], model.mg_relw[mg_data["total_edge_type"]])
        
        # Compute predictions for PPI layers
        ppi_preds = dict()
        for celltype, x in ppi_x.items():
            ppi_preds[celltype] = el_dot(x.to(device), ppi_data_batch[celltype]['total_edge_index'].to(device), [])
            ppi_preds_all[celltype] = torch.cat([ppi_preds_all.setdefault(celltype, torch.tensor([])), ppi_preds[celltype].detach().cpu()])
            ppi_data_y[celltype]['y'] = torch.cat([ppi_data_y[celltype]['y'], ppi_data_batch[celltype]['y'].detach().cpu()])
            ppi_data_y[celltype]['total_edge_type'] = torch.cat([ppi_data_y[celltype]['total_edge_type'], ppi_data_batch[celltype]['total_edge_type'].detach().cpu()])

    return ppi_x, mg_x, mg_pred, ppi_preds_all, ppi_data_y
    

def construct_batch_center_loss_mask(original_mask: list, ppi_node_ind_batch: dict, ppi_x_ori: dict) -> list:
    """
    Construct the center loss mask for a batch of nodes during training. We first concatenate batched indices across cell types. Note that we must add each index a certain amount correponding to its cell type to represent its position in the concatenated list. We then find the intersection between the indices and the original train mask. The POSITIONS of the matching concatenated batched indices are the mask we want.
    
    :param original_mask: The original train mask for nodes for calculating center_loss.
    :param ppi_node_ind_batch: A dictionary of ppi node indices that are sampled in this minibatch.
    :param ppi_x_ori: A dictionary of original :code:`ppi_x`. We need it in order to know the original numbers of PPI nodes of each cell type, and the order of cell types in the original train mask.
    
    :return: A list of new train mask for center loss that can be directly applied on the batch center loss labels and the batch labels and the batch embeddings.
        
    """
    ppi_size = [x.shape[0] for x in ppi_x_ori.values()]
    ppi_cum_size = np.append(0, np.cumsum(ppi_size))
    ppi_node_ind_batch_concat = torch.cat([value + ppi_cum_size[i] for i, value in enumerate(ppi_node_ind_batch.values())]).detach().cpu().numpy()
    original_mask_set = set(original_mask)
    train_mask_batch = [i for i, ind in enumerate(ppi_node_ind_batch_concat) if ind in original_mask_set]
    
    return train_mask_batch
    

def train_batch2dict(packed_batch: object, mg_x_ori: dict, ppi_metapaths: dict, cell_type_order: list, device: str) -> dict:
    """
    Re-initialize :code:`metagraph`, and transform packed batches of all graphs to a dictionary of batches.
    
    :param packed_batch: An iterable (tuple if directly following unpacking of the output of :code:`generatePPIBatch`) storing batches of :class:`Data` from all graphs in one round.
    :param mg_x_ori: metagraph original node embeddings.
    :param ppi_metapaths: All metapaths.
    :param cell_type_order: Cell type order.
    :param device: A string indicating the device. Default is "cuda".
    
    :return: A dictionary of edge data from all graphs in one round, :code:`ppi_x_batch`, :code:`ppi_node_ind_batch` and :code:`ppi_metapaths_batch` extracted from batches, and the re-initialized node embeddings :code:`mg_x_init`.
    """
    # Re-initalize mg_x from mg_x in each batch
    # ppi_x_init = {key:x.clone().to(device) for key, x in ppi_x.items()}
    # mg_x_init = mg_x_ori.clone().to(device) if len(mg_x_ori)!=0 else []
    
    # Unpack batches
    ppi_data_batch = {key:{} for key in cell_type_order}
    ppi_x_batch = {}
    ppi_node_ind_batch = {}
    ppi_metapaths_out = {}
    for ind, batch in enumerate(packed_batch):
        # ori_map = batch.n_id
        # batch.total_edge_index = ori_map[batch.edge_index]
        i = cell_type_order[ind]
        ppi_node_ind_batch[i] = batch.n_id.to(device)
        ppi_x_batch[i] = batch.x.to(device)
        ppi_data_batch[i]['total_edge_index'] = batch.edge_index.to(device)
        ppi_data_batch[i]['total_edge_type'] = batch.edge_attr.to(device)
        ppi_data_batch[i]['y'] = batch.y.to(device)
        
        # Metapath adjs
        ppi_metapaths_batch = construct_metapath(ppi_metapaths, batch.edge_index[:, batch.y.type(torch.bool)], batch.edge_attr[batch.y.type(torch.bool)], batch.x.shape[0])
        
        ppi_metapaths_out[i] = [ppi_metapaths_batch[0].to(device)]
    
    return ppi_data_batch, ppi_x_batch, ppi_node_ind_batch, ppi_metapaths_out, []#, mg_x_init


def generate_batch(data_dict, metapaths, edge_attr_dict, mask, batch_size, device, ppi=False, loader_type="graphsaint", num_layers=2):
    masked_data_dict = dict()
    metapath_adjs_dict = dict()
    x_dict = dict()
    loader_dict = {}

    # Iterate through subnetworks
    for key, data in data_dict.items():
        
        # Positive edges
        if mask == "train":
            pos_edge_index = data.edge_index[:, data.train_mask]
            edge_type = data.edge_attr[data.train_mask]
        elif mask == "val":
            pos_edge_index = data.edge_index[:, data.val_mask] 
            edge_type = data.edge_attr[data.val_mask] 
        elif mask == "test":
            pos_edge_index = data.edge_index[:, data.test_mask] 
            edge_type = data.edge_attr[data.test_mask]
        else:
            pos_edge_index = data.edge_index
            edge_type = data.edge_attr
        
        # Negative edges
        neg_edge_index, neg_edge_type = negative_sampler(pos_edge_index, edge_type, edge_attr_dict)
        
        # All edges and labels
        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1) 
        total_edge_type = torch.cat([edge_type, neg_edge_type], dim=-1)
        y = torch.zeros(total_edge_index.size(1)).float() 
        y[:pos_edge_index.size(1)] = 1

        # Metapath adjs
        metapath_adjs_dict[key] = construct_metapath(metapaths, pos_edge_index, edge_type, data.x.size(0))

        # Save information
        x_dict[key] = data.x.to(device)
        masked_data_dict[key] = dict()
        if ppi:
            data = Data(x = data.x, edge_index = total_edge_index, edge_attr = total_edge_type, y = y)
            data.n_id = torch.arange(data.num_nodes)
            if loader_type == "neighbor":
                loader = NeighborLoader(data, num_neighbors = [-1] * num_layers, batch_size = batch_size, input_nodes = torch.arange(data.num_nodes), shuffle = True)
            elif loader_type == "graphsaint":
                #loader = GraphSAINTRandomWalkSampler(data, batch_size = batch_size, walk_length = num_layers)
                loader = GraphSAINTEdgeSampler(data, batch_size = batch_size, num_steps = 16)
            else:
                raise NotImplementedError

            loader_dict[key] = loader
            masked_data_dict[key]["total_edge_type"] = total_edge_type
        else:
            masked_data_dict[key]["total_edge_index"] = total_edge_index.to(device)
            masked_data_dict[key]["total_edge_type"] = total_edge_type.to(device)
            masked_data_dict[key]["y"] = y.to(device)

    return loader_dict, masked_data_dict, metapath_adjs_dict, x_dict


def negative_sampler(pos_edge_index, edge_type, edge_attr_dict):
    if len(edge_type) == 0: return pos_edge_index, edge_type
    neg_edge_index = None 
    neg_edge_type = []
    for attr, idx in edge_attr_dict.items():
        mask = (edge_type == idx)
        if mask.sum() == 0: continue
        pos_rel_edge_index = pos_edge_index.T[mask].T        
        neg_source, neg_target, neg_rand = structured_negative_sampling(pos_rel_edge_index)
        neg_rel_edge_index = torch.stack((neg_source, neg_rand), dim = 0)
        """
        neg_rel_edge_index = pos_rel_edge_index.clone()
        rand_axis = random.sample([0, 1], 1)[0]
        rand_index = torch.randperm(pos_rel_edge_index.size(1))
        neg_rel_edge_index[rand_axis, :] = pos_rel_edge_index[rand_axis, rand_index]
        """
        if neg_edge_index == None: neg_edge_index = neg_rel_edge_index 
        else: neg_edge_index = torch.cat((neg_edge_index, neg_rel_edge_index), 1) 
        
        neg_edge_type.extend([idx] * mask.sum())

    return neg_edge_index, torch.tensor(neg_edge_type)