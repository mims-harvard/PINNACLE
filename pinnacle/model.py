import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, LayerNorm

from conv import PCTConv, PPIConv


class Pinnacle(nn.Module):
    def __init__(self, nfeat, hidden, output, num_ppi_relations, num_mg_relations, ppi_data, n_heads, pc_att_channels, dropout = 0.2):
        super(Pinnacle, self).__init__()

        self.dropout = dropout

        # Layer dimensions
        self.layer1_in = nfeat
        self.layer1_out = hidden
        self.layer2_in = self.layer1_out * n_heads
        self.layer2_out = output
        self.output = self.layer2_out * n_heads

        # Complete layer #1
        self.conv1_up = PCTConv(self.layer1_in, num_ppi_relations, num_mg_relations, ppi_data, self.layer1_out, sem_att_channels=8, pc_att_channels=pc_att_channels, node_heads=n_heads)
        self.conv1_down = PPIConv(self.layer1_out * n_heads, num_ppi_relations, self.layer1_out, ppi_data, sem_att_channels=8, node_heads=n_heads)

        # Normalization
        self.layer_norm1 = LayerNorm(self.layer2_in)
        self.batch_norm1 = BatchNorm(self.layer2_in)

        # Complete layer #2
        self.conv2_up = PCTConv(self.layer2_in, num_ppi_relations, num_mg_relations, ppi_data, self.layer2_out, sem_att_channels=8, pc_att_channels=pc_att_channels, node_heads=n_heads)
        self.conv2_down = PPIConv(self.layer2_out * n_heads, num_ppi_relations, self.layer2_out, ppi_data, sem_att_channels=8, node_heads=n_heads)

        # Metagraph decoder
        self.mg_relw = nn.Parameter(torch.Tensor(num_mg_relations, int(self.output)))
        nn.init.xavier_uniform_(self.mg_relw, gain = nn.init.calculate_gain('leaky_relu'))


    def forward(self, ppi_x, mg_x, ppi_metapaths, mg_metapaths, ppi_edge_index, mg_edge_index, tissue_neighbors):
        
        ########################################
        # Complete layer #1
        ########################################

        # Update Protein-Celltype-Tissue
        ppi_x, mg_x = self.conv1_up(ppi_x, mg_x, ppi_metapaths, mg_metapaths, ppi_edge_index, mg_edge_index, tissue_neighbors, init_cci=True)

        # Update PPI and down-pool metagraph
        ppi_x = self.conv1_down(ppi_x, ppi_metapaths, mg_x, self.conv1_up.ppi_attn)

        ########################################
        # Apply Leaky ReLU, dropout, and normalize
        ########################################
        for celltype, x in ppi_x.items():
            ppi_x[celltype] = self.layer_norm1(x)
            ppi_x[celltype] = F.leaky_relu(ppi_x[celltype])
            ppi_x[celltype] = self.batch_norm1(ppi_x[celltype])
            ppi_x[celltype] = F.dropout(ppi_x[celltype], p = self.dropout, training = self.training)
        mg_x = self.layer_norm1(mg_x)
        mg_x = F.leaky_relu(mg_x)
        mg_x = self.batch_norm1(mg_x)
        mg_x = F.dropout(mg_x, p = self.dropout, training = self.training)

        ########################################
        # Complete layer #2
        ########################################

        # Update Protein-Celltype-Tissue
        ppi_x, mg_x = self.conv2_up(ppi_x, mg_x, ppi_metapaths, mg_metapaths, ppi_edge_index, mg_edge_index, tissue_neighbors)

        # Update PPI and down-pool metagraph
        ppi_x = self.conv2_down(ppi_x, ppi_metapaths, mg_x, self.conv2_up.ppi_attn)

        return ppi_x, mg_x
