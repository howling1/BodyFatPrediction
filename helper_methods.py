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

def evaluate(model, loader, device, config):
    """
    Function for validation
    :param model: initialized model
    :param loader: data loader
    :param device: torch device
    :param config: model config
    """
    model.eval()
 
    crit = torch.nn.MSELoss() if config['task'] == "regression" else torch.nn.CrossEntropyLoss()
    predictions = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data) if config['task'] == "regression" else F.softmax(model(data), dim=1)
            target = data.y.reshape((-1, config["num_classes"]))
            predictions = torch.cat((predictions, pred))
            targets = torch.cat((targets, target))

    loss = crit(predictions, targets)
    acc = r2_score(targets.detach().cpu(), predictions.detach().cpu(), multioutput='raw_values') if config['task'] == "regression" else np.mean((torch.argmax(predictions,1)==torch.argmax(targets,1)).numpy())

    return loss, acc

def load_and_split_dataset(raw_data_root, dataset_root, basic_features_path, ids_root, target):
    """
    Wrapper function for IMDataset
    :param raw_data_root: path for the input mesh data 
    :param dataset_root: path where the in memory dataset will be written
    :param basic_features_path: path of the "basic_features.csv"
    :param target: name of the feature to be predicted
    """
    dataset_female = IMDataset(raw_data_root, dataset_root, basic_features_path, ids_root, target, 0)
    dataset_male = IMDataset(raw_data_root, dataset_root, basic_features_path, ids_root, target, 1)

    train_data_female, dev_data_female = train_test_split(dataset_female, test_size=0.4, random_state=42)
    val_data_female, test_data_female = train_test_split(dev_data_female, test_size=0.5, random_state=42)
    train_data_male, dev_data_male = train_test_split(dataset_male, test_size=0.4, random_state=42)
    val_data_male, test_data_male = train_test_split(dev_data_male, test_size=0.5, random_state=42)

    train_data_all = train_data_male + train_data_female
    val_data_all = val_data_male + val_data_female
    test_data_all = test_data_male + test_data_female
    
    random.shuffle(train_data_all)
    random.shuffle(val_data_all)
    random.shuffle(test_data_all)

    # return train_data_all, val_data_all, test_data_all
    return train_data_all, val_data_all, test_data_male, test_data_female



    
    