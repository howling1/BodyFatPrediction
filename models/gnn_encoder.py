import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils.dropout import dropout_edge

from helper_methods import get_conv_layers

class GraphFeatureEncoder(torch.nn.Module):
    """Graph neural network consisting of stacked graph convolutions."""
    def __init__(
        self,
        gnn_conv,
        in_features,
        conv_channels,
        apply_dropedge,
        apply_bn,
        num_heads=None,
    ):
        super().__init__()

        self.apply_dropedge = apply_dropedge
        self.apply_bn = apply_bn

        *first_conv_channels, final_conv_channel = conv_channels
        conv_layers = get_conv_layers(
            channels=[in_features] + conv_channels,
            conv=gnn_conv,
            num_heads=num_heads,
        )
        
        self.conv_layers = nn.ModuleList(conv_layers)

        self.bn_layers = [None for _ in first_conv_channels]

        if apply_bn:
            if gnn_conv == GATConv:
                self.bn_layers = nn.ModuleList(
                    [nn.BatchNorm1d(channel * num_heads) for channel in first_conv_channels]
                )
            else:  
                self.bn_layers = nn.ModuleList(
                    [nn.BatchNorm1d(channel) for channel in first_conv_channels]
                )

    def forward(self, x, edge_index):
        *first_conv_layers, final_conv_layer = self.conv_layers

        for conv_layer, bn_layer in zip(first_conv_layers, self.bn_layers):
            if self.training and self.apply_dropedge:
                edge_index, _ = dropout_edge(edge_index)

            x = conv_layer(x, edge_index) 

            if bn_layer is not None:
                x = bn_layer(x)

            x = F.relu(x)

        return final_conv_layer(x, edge_index)

