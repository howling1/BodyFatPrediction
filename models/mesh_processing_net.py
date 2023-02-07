import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max
from torch_geometric.nn import GATConv

from helper_methods import get_mlp_layers
from models.gnn_encoder import GraphFeatureEncoder

class MeshProcessingNetwork(torch.nn.Module):
    """Mesh processing network."""
    def __init__(
        self,
        gnn_conv,
        in_features, # features of node: x, y, z
        encoder_channels, # features of the linear encoder
        conv_channels, # features of conv
        decoder_channels, # features of the prediction MLP
        num_classes,
        apply_dropedge, 
        apply_bn,
        apply_dropout,
        aggregation, # mean or max
        num_heads=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.aggregation = aggregation


        encoder_channels = [in_features] + encoder_channels
        self.input_encoder = get_mlp_layers(
            channels=encoder_channels,
            activation=nn.ReLU,
            output_activation=nn.ReLU,
            apply_dropout=apply_dropout
        )
        self.gnn = GraphFeatureEncoder(
            gnn_conv=gnn_conv,
            in_features=encoder_channels[-1],
            conv_channels=conv_channels,
            num_heads=num_heads,
            apply_dropedge=apply_dropedge,
            apply_bn=apply_bn,
        )
        *_, final_conv_channel = conv_channels

        decoder_in = final_conv_channel * num_heads if gnn_conv == GATConv else final_conv_channel
        
        self.final_projection = get_mlp_layers(
            [decoder_in] + decoder_channels + [num_classes],
            activation=nn.ReLU,
            apply_dropout=apply_dropout
        )


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_encoder(x) if self.input_encoder != None else x
        x = self.gnn(x, edge_index)
        x = scatter_mean(x, batch, dim=0) if self.aggregation == 'mean' else scatter_max(x, batch, dim=0)[0]
        x = self.final_projection(x)

        return torch.squeeze(x, 1) if self.num_classes == 1 else x
    

