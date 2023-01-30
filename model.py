import torch
import torch.nn as nn
import math
from torch_scatter import scatter_mean, scatter_max
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, MessagePassing, JumpingKnowledge
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.utils.dropout import dropout_edge, dropout_node, dropout_path
from itertools import tee


def pairwise(iterable):
    """Iterate over all pairs of consecutive items in a list.
    Notes
    -----
        [s0, s1, s2, s3, ...] -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def get_conv_layers(channels: list, conv, num_heads=None, is_dense=False):
    """Define convolution layers with specified in and out channels.

    Parameters
    ----------
    channels: list
        List of integers specifying the size of the convolution channels.
    conv: GNN Convolution layer.
    conv_params: dict
        Dictionary specifying convolution parameters.

    Returns
    -------
    list
        List of convolutions with the specified channels.
    """

    if is_dense:
        if conv == GATConv:
            conv_layers = [
            conv(in_ch, out_ch, num_heads) if i == 0 else conv(in_ch * num_heads * int(math.pow(2, i - 1)), out_ch, num_heads) for i, (in_ch, out_ch) in enumerate(pairwise(channels))
            ]
        else:
            conv_layers = [
            conv(in_ch, out_ch) if i == 0 else conv(in_ch * int(math.pow(2, i - 1)), out_ch) for i,(in_ch, out_ch) in enumerate(pairwise(channels))
            ]

    else:
        if conv == GATConv:
            conv_layers = [
            conv(in_out[0], in_out[1], num_heads) if i == 0 else conv(in_out[0] * num_heads, in_out[1], num_heads) for i, in_out in enumerate(pairwise(channels))
            ]
        else:
            conv_layers = [
            conv(in_ch, out_ch) for in_ch, out_ch in pairwise(channels)
            ]

    return conv_layers

def get_mlp_layers(channels: list, activation, output_activation=nn.Identity, apply_dropout=False):
    """Define basic multilayered perceptron network."""
    if len(channels) == 1:
        return None

    layers = []
    *intermediate_layer_definitions, final_layer_definition = pairwise(channels)

    for in_ch, out_ch in intermediate_layer_definitions:
        intermediate_layer = nn.Linear(in_ch, out_ch)
        layers += [intermediate_layer, activation()]
        if apply_dropout:
            layers.append(nn.Dropout())

    layers += [nn.Linear(*final_layer_definition), output_activation()]

    return nn.Sequential(*layers)

class FeatureSteeredConvolution(MessagePassing):
    """Implementation of feature steered convolutions.

    References
    ----------
    .. [1] Verma, Nitika, Edmond Boyer, and Jakob Verbeek.
       "Feastnet: Feature-steered graph convolutions for 3d shape analysis."
       Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        ensure_trans_invar: bool = True,
        bias: bool = True,
        with_self_loops: bool = True,
    ):
        super().__init__(aggr="mean")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.with_self_loops = with_self_loops

        self.linear = torch.nn.Linear(
            in_features=in_channels,
            out_features=out_channels * num_heads,
            bias=False,
        )
        self.u = torch.nn.Linear(
            in_features=in_channels,
            out_features=num_heads,
            bias=False,
        )
        self.c = torch.nn.Parameter(torch.Tensor(num_heads))

        if not ensure_trans_invar:
            self.v = torch.nn.Linear(
                in_features=in_channels,
                out_features=num_heads,
                bias=False,
            )
        else:
            self.register_parameter("v", None)

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialization of tuneable network parameters."""
        torch.nn.init.uniform_(self.linear.weight)
        torch.nn.init.uniform_(self.u.weight)
        torch.nn.init.normal_(self.c, mean=0.0, std=0.1)
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0.0, std=0.1)
        if self.v is not None:
            torch.nn.init.uniform_(self.v.weight)

    def forward(self, x, edge_index):
        """Forward pass through a feature steered convolution layer.

        Parameters
        ----------
        x: torch.tensor [|V|, in_features]
            Input feature matrix, where each row describes
            the input feature descriptor of a node in the graph.
        edge_index: torch.tensor [2, E]
            Edge matrix capturing the graph's
            edge structure, where each row describes an edge
            between two nodes in the graph.
        Returns
        -------
        torch.tensor [|V|, out_features]
            Output feature matrix, where each row corresponds
            to the updated feature descriptor of a node in the graph.
        """
        if self.with_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index = edge_index, num_nodes = x.shape[0])

        out = self.propagate(edge_index, x=x)
        
        return out if self.bias is None else out + self.bias

    def _compute_attention_weights(self, x_i, x_j):
        """Computation of attention weights.

        Parameters
        ----------
        x_i: torch.tensor [|E|, in_feature]
            Matrix of feature embeddings for all central nodes,
            collecting neighboring information to update its embedding.
        x_j: torch.tensor [|E|, in_features]
            Matrix of feature embeddings for all neighboring nodes
            passing their messages to the central node along
            their respective edge.
        Returns
        -------
        torch.tensor [|E|, M]
            Matrix of attention scores, where each row captures
            the attention weights of transformed node in the graph.
        """
        if x_j.shape[-1] != self.in_channels:
            raise ValueError(
                f"Expected input features with {self.in_channels} channels."
                f" Instead received features with {x_j.shape[-1]} channels."
            )
        if self.v is None:
            attention_logits = self.u(x_i - x_j) + self.c
        else:
            attention_logits = self.u(x_i) + self.b(x_j) + self.c
        return F.softmax(attention_logits, dim=1)

    def message(self, x_i, x_j):
        """Message computation for all nodes in the graph.

        Parameters
        ----------
        x_i: torch.tensor [|E|, in_feature]
            Matrix of feature embeddings for all central nodes,
            collecting neighboring information to update its embedding.
        x_j: torch.tensor [|E|, in_features]
            Matrix of feature embeddings for all neighboring nodes
            passing their messages to the central node along
            their respective edge.
        Returns
        -------
        torch.tensor [|E|, out_features]
            Matrix of updated feature embeddings for
            all nodes in the graph.
        """
        attention_weights = self._compute_attention_weights(x_i, x_j)
        x_j = self.linear(x_j).view(-1, self.num_heads, self.out_channels)
        return (attention_weights.view(-1, self.num_heads, 1) * x_j).sum(dim=1)

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
        num_heads=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_encoder = get_mlp_layers(
            channels=[in_features] + encoder_channels,
            activation=nn.ReLU,
            output_activation=nn.ReLU,
            apply_dropout=apply_dropout
        )
        self.gnn = GraphFeatureEncoder(
            gnn_conv=gnn_conv,
            in_features=encoder_channels[-1] if len(encoder_channels) > 0 else in_features,
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
        x = scatter_mean(x, batch, dim=0)
        x = self.final_projection(x)

        return torch.squeeze(x, 1) if self.num_classes == 1 else x
    

class ResGNN(torch.nn.Module):
        def __init__(
                self, 
                gnn_conv, 
                in_features, 
                inout_conv_channel: int, 
                num_layers: int,
                num_skip_layers: int,
                apply_dropedge: bool, 
                apply_bn: bool,
                apply_dropout: bool,
                encoder_channels: list,
                aggregation: str, # mean, max
                decoder_channels: list, 
                num_classes: int,
                num_heads: int = None
                ):
            super(ResGNN, self).__init__()

            self.num_classes = num_classes
            self.apply_dropedge = apply_dropedge
            self.num_layers = num_layers
            self.num_skip_layers = num_skip_layers
            self.aggregation = aggregation

            conv_channels = [inout_conv_channel for i in range(num_layers)]
            conv_layers = get_conv_layers(
                channels=[in_features] + conv_channels,
                conv=gnn_conv,
                num_heads=num_heads
            )

            self.conv_layers = nn.ModuleList(conv_layers)
            self.bn_layers = [None for _ in range(num_layers)]

            if apply_bn:
                self.bn_layers = nn.ModuleList(
                    [nn.BatchNorm1d(inout_conv_channel * num_heads) for i in range(num_layers)] if gnn_conv == GATConv 
                    else [nn.BatchNorm1d(inout_conv_channel) for i in range(num_layers)]
                    )

            conv_out_channel = inout_conv_channel * num_heads * num_layers if gnn_conv == GATConv else inout_conv_channel * num_layers

            self.linear_encoder = get_mlp_layers(
            [conv_out_channel] + encoder_channels,
            activation=nn.ReLU,
            output_activation=nn.ReLU,
            )

            global_features_channel = encoder_channels[-1]

            self.final_projection = get_mlp_layers(
            [global_features_channel] + decoder_channels + [num_classes],
            activation=nn.ReLU,
            apply_dropout=apply_dropout
        )
        
        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch

            previous_outputs = []

            for i, (conv_layer, bn_layer) in enumerate(zip(self.conv_layers, self.bn_layers)):
                if self.training and self.apply_dropedge:
                    edge_index, _ = dropout_edge(edge_index)

                x = conv_layer(x, edge_index)

                if bn_layer is not None:
                    x = bn_layer(x)
                    
                x = F.relu(x)

                if i % self.num_skip_layers == 0 and i != 0 and i != self.num_layers - 1:
                    x += previous_outputs[i - self.num_skip_layers]

                previous_outputs.append(x)

            
            local_features = torch.cat(previous_outputs, dim=1)
            global_feature = scatter_mean(self.linear_encoder(local_features), batch, dim=0) if self.aggregation == 'mean' else scatter_max(self.linear_encoder(local_features), batch, dim=0)[0]
            output = self.final_projection(global_feature)

            return torch.squeeze(output, 1) if self.num_classes == 1 else output
            
class DenseGNN(torch.nn.Module):
        def __init__(
                self, 
                gnn_conv, 
                in_features, 
                inout_conv_channel: int, 
                num_layers: int,
                apply_dropedge: bool, 
                apply_bn: bool,
                apply_dropout: bool,
                encoder_channels: list,
                aggregation: str, # mean, max
                decoder_channels: list, 
                num_classes: int,
                num_heads: int = None
                ):
            super(DenseGNN, self).__init__()

            self.num_classes = num_classes
            self.apply_dropedge = apply_dropedge
            self.num_layers = num_layers
            self.aggregation = aggregation

            conv_channels = [inout_conv_channel for i in range(num_layers)]
            conv_layers = get_conv_layers(
                channels=[in_features] + conv_channels,
                conv=gnn_conv,
                num_heads=num_heads,
                is_dense=True
            )

            # for conv_layer in conv_layers


            self.conv_layers = nn.ModuleList(conv_layers)
            self.bn_layers = [None for _ in range(num_layers)]

            if apply_bn:
                self.bn_layers = nn.ModuleList(
                    [nn.BatchNorm1d(inout_conv_channel * num_heads) for i in range(num_layers)] if gnn_conv == GATConv 
                    else [nn.BatchNorm1d(inout_conv_channel) for i in range(num_layers)]
                    )

            conv_out_channel = int(math.pow(2, num_layers - 1)) * inout_conv_channel * num_heads if gnn_conv == GATConv else int(math.pow(2, num_layers - 1)) * inout_conv_channel

            self.linear_encoder = get_mlp_layers(
            [conv_out_channel] + encoder_channels,
            activation=nn.ReLU,
            output_activation=nn.ReLU,
            )

            global_features_channel = encoder_channels[-1]

            self.final_projection = get_mlp_layers(
            [global_features_channel] + decoder_channels + [num_classes],
            activation=nn.ReLU,
            apply_dropout=apply_dropout
            )
        
        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch

            # print(self.conv_layers)

            previous_outputs = []

            for i, (conv_layer, bn_layer) in enumerate(zip(self.conv_layers, self.bn_layers)):
                if self.training and self.apply_dropedge:
                    edge_index, _ = dropout_edge(edge_index)

                x = conv_layer(x, edge_index)

                if bn_layer is not None:
                    x = bn_layer(x)
                    
                x = F.relu(x)

                x = torch.cat(previous_outputs + [x], dim=1)

                previous_outputs.append(x)

            global_feature = scatter_mean(self.linear_encoder(x), batch, dim=0) if self.aggregation == 'mean' else scatter_max(self.linear_encoder(x), batch, dim=0)[0]
            output = self.final_projection(global_feature)

            return torch.squeeze(output, 1) if self.num_classes == 1 else output

class JKNet(nn.Module):
    def __init__(
            self,
            in_features, 
            num_classes, 
            gnn_conv, # GAT not supported for now
            mode='max', 
            num_layers=6, 
            hidden=16
            ):
        
        super(JKNet, self).__init__()
        self.num_layers = num_layers
        self.mode = mode
        self.conv0 = gnn_conv(in_features, hidden)
        self.bn0 = nn.BatchNorm1d(hidden)
        # self.dropout0 = nn.Dropout(p=0.5)
        self.num_classes = num_classes

        for i in range(1, self.num_layers):
            setattr(self, 'conv{}'.format(i), gnn_conv(hidden, hidden))
            setattr(self, 'bn{}'.format(i), nn.BatchNorm1d(hidden))
            # setattr(self, 'dropout{}'.format(i), nn.Dropout(p=0.5))

        self.jk = JumpingKnowledge(mode=mode)
        if mode == 'max':
            self.fc = nn.Linear(hidden, num_classes)
        elif mode == 'cat':
            self.fc = nn.Linear(num_layers * hidden, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        layer_out = []  #save the result of every layer
        for i in range(self.num_layers):
            conv = getattr(self, 'conv{}'.format(i))
            bn = getattr(self, 'bn{}'.format(i))
            # dropout = getattr(self, 'dropout{}'.format(i))
            if self.training:
                edge_index, _ = dropout_edge(edge_index)
            x = F.relu(bn(conv(x, edge_index)))
            layer_out.append(x)

        h = self.jk(layer_out)  # JK layer

        h = scatter_mean(h, batch, dim=0)

        h = self.fc(h)


        return torch.squeeze(h, 1) if self.num_classes == 1 else h
            