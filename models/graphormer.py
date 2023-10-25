import torch
import torch.nn as nn

from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_adj, get_laplacian
from helper_methods import get_mlp_layers
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Graphormer(nn.Module):
    """Graph transformer. Can handle batch graphs with the same number of nodes"""
    def __init__(self, dim_node, hidden_dim, n_targets, n_layers, n_heads, decoder_channels, dropout, device):
        super().__init__()
        self.device = device
        self.n_heads = n_heads
        self.input_dim = dim_node * n_heads
        encoder_layers = TransformerEncoderLayer(self.input_dim, n_heads, hidden_dim, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        self.final_projection = get_mlp_layers(
            [self.input_dim] + decoder_channels + [n_targets],
            activation=nn.ReLU,
        )

    def create_adjacent_matrices(self, batch_edges, n_nodes):
        A = []

        for i in range(len(batch_edges)):
            a = to_dense_adj(edge_index=batch_edges[i], max_num_nodes=n_nodes)[0]
            A.append(a)

        A = torch.stack(A, dim=0)
        
        return A

    def max_without_degree_zero(self, batch_nodes, batch_adjacent_matrices):
        batch_degree_matrices = torch.sum(batch_adjacent_matrices, dim=1)
        mask = batch_degree_matrices != 0
        batch_nodes[mask == False] = float('-inf')

        return torch.max(batch_nodes, dim=1)[0]

    def split_batch(self, batch_data):
        data_list = Batch.to_data_list(batch_data)
        
        batch_nodes = []
        batch_edges = []

        for d in data_list:
            batch_nodes.append(d.x)
            batch_edges.append(d.edge_index)

        batch_nodes = torch.stack(batch_nodes, dim=0)

        return batch_nodes, batch_edges

    def forward(self, data):
        batch_nodes, batch_edges = self.split_batch(data)
        n_nodes = batch_nodes[0].shape[0]

        # TODO: add some potential posistional embedding to the node embedding for optimization
        
        # concatenate nodes for multi-head attention computation
        batch_multi_nodes = torch.cat([batch_nodes] * self.n_heads, dim=2)
        
        A = self.create_adjacent_matrices(batch_edges, n_nodes).to(self.device)
        masks = torch.cat([A] * self.n_heads, dim=0)

        output = self.transformer_encoder(src=batch_multi_nodes, mask=masks)

        batch_adjacent_matrices = masks[0: batch_nodes.shape[0]]
        output = self.max_without_degree_zero(output, batch_adjacent_matrices)

        output = self.final_projection(output)

        return output
