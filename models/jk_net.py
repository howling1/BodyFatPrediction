import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max
import torch.nn.functional as F
from torch_geometric.nn import GATConv, JumpingKnowledge
from torch_geometric.utils.dropout import dropout_edge

from helper_methods import get_conv_layers, get_mlp_layers

class JKNet(nn.Module):
    def __init__(
            self,
            gnn_conv,
            num_hiddens,
            in_features, 
            num_layers,
            num_classes, 
            jk_mode, # cat, mean, lstm
            apply_dropedge: bool, 
            apply_bn: bool,
            apply_dropout: bool,
            encoder_channels: list,
            aggregation: str, # mean, max
            decoder_channels: list, 
            num_heads: int = None # only applicable for GAT
            ):
        
        super(JKNet, self).__init__()
        self.num_layers = num_layers
        self.jk_mode = jk_mode
        self.num_classes = num_classes
        self.aggregation = aggregation
        self.apply_bn = apply_bn
        self.apply_dropedge = apply_dropedge
        self.num_heads = num_heads

        conv_channels = [num_hiddens for i in range(num_layers)]
        conv_layers = get_conv_layers(
            channels=[in_features] + conv_channels,
            conv=gnn_conv,
            num_heads=num_heads,
        )

        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = [None for _ in range(num_layers)]

        if apply_bn:
            self.bn_layers = nn.ModuleList(
                [nn.BatchNorm1d(num_hiddens * num_heads) for i in range(num_layers)] if gnn_conv == GATConv 
                else [nn.BatchNorm1d(num_hiddens) for i in range(num_layers)]
                )

        if jk_mode == 'lstm':
            self.jk = JumpingKnowledge(jk_mode, num_hiddens * num_heads, num_layers) if gnn_conv == GATConv else JumpingKnowledge(jk_mode, num_hiddens, num_layers)
        else: 
            self.jk = JumpingKnowledge(jk_mode)

        if jk_mode == 'max' or jk_mode == 'lstm':
            encoder_in = num_hiddens * num_heads if gnn_conv == GATConv else num_hiddens
        elif jk_mode == 'cat':
            encoder_in = num_layers * num_hiddens * num_heads if gnn_conv == GATConv else num_hiddens * num_layers

        encoder_channels = [encoder_in] + encoder_channels

        self.linear_encoder = get_mlp_layers(
            encoder_channels,
            activation=nn.ReLU,
            output_activation=nn.ReLU,
            )

        decoder_in = encoder_channels[-1]

        self.final_projection = get_mlp_layers(
            [decoder_in] + decoder_channels + [num_classes],
            activation=nn.ReLU,
            apply_dropout=apply_dropout
            )

        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        layer_out = []  #save the result of every layer
        for conv_layer, bn_layer in zip(self.conv_layers, self.bn_layers):
            if self.training and self.apply_dropedge:
                edge_index, _ = dropout_edge(edge_index)

            x = conv_layer(x, edge_index)

            if bn_layer is not None:
                x = bn_layer(x)
                
            x = F.relu(x)
            layer_out.append(x)

        x = self.jk(layer_out)  # JK layer

        global_feature = scatter_mean(self.linear_encoder(x), batch, dim=0) if self.aggregation == 'mean' else scatter_max(self.linear_encoder(x), batch, dim=0)[0]
        output = self.final_projection(global_feature)

        return torch.squeeze(output, 1) if self.num_classes == 1 else output