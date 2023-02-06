import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
import random
from itertools import tee
from torch_geometric.nn import GATConv
from models.feast_conv import FeatureSteeredConvolution
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from datasets.in_memory import IMDataset

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
        elif conv == FeatureSteeredConvolution:
            conv_layers = [
            conv(in_ch, out_ch, num_heads) for in_ch, out_ch in pairwise(channels)
            ]    
        else:
            conv_layers = [
            conv(in_ch, out_ch) for in_ch, out_ch in pairwise(channels)
            ]

    return conv_layers

def get_mlp_layers(channels: list, activation=nn.ReLU, output_activation=nn.Identity, apply_dropout=False):
    """Define basic multilayered perceptron network."""
    if len(channels) == 1:
        return nn.Sequential(nn.Identity())

    layers = []
    *intermediate_layer_definitions, final_layer_definition = pairwise(channels)

    for in_ch, out_ch in intermediate_layer_definitions:
        intermediate_layer = nn.Linear(in_ch, out_ch)
        layers += [intermediate_layer, activation()]
        if apply_dropout:
            layers.append(nn.Dropout())

    layers += [nn.Linear(*final_layer_definition), output_activation()]

    return nn.Sequential(*layers)

def evaluate(model, loader, device, task):
    model.eval()
 
    crit = torch.nn.MSELoss() if task == "regression" else torch.nn.CrossEntropyLoss()
    predictions = torch.tensor([])
    targets = torch.tensor([])
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data).detach().cpu() if task == "regression" else F.softmax(model(data).detach().cpu(), dim=1)
            target = data.y.detach().cpu()
            predictions = torch.cat((predictions, pred))
            targets = torch.cat((targets, target))

    loss = crit(predictions, targets)
    acc = r2_score(targets, predictions) if task == "regression" else np.mean((torch.argmax(predictions,1)==torch.argmax(targets,1)).numpy())

    return loss, acc