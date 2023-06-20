import numpy as np
import pandas as pd
import torch.nn.functional as F
import wandb
import torch
import random
from tqdm import tqdm
from sklearn.metrics import r2_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from models.mesh_processing_net import MeshProcessingNetwork
from helper_methods import evaluate, load_and_split_dataset
from models.shrinkage_loss import RegressionShrinkageLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from datasets.in_memory import IMDataset
from sklearn.model_selection import train_test_split, KFold


if __name__ == "__main__":
    config = {
    "experiment_name" : "cv_all_sage_5k", # there should be a folder named exactly this under the folder runs/
    "batch_size" : 32,
    "epochs" : 100,
    "base_lr" : 0.001,
    "decayed_lr": True,
    "weight_decay": 0.,
    "clip_norm": 1,
    "num_classes": 2,
    "task" : "regression", # "regression" or "classification"
    "print_every_n" : 1000,
    "validate_every_n" : 1000}
 
    # n_class = 1 if config["task"] == "regression" else 2

# template for  MeshProcressingNet params
    model_params = dict(
        gnn_conv = SAGEConv,
        in_features = 3,
        encoder_channels = [],
        conv_channels = [32, 64, 128],
        decoder_channels = [512, 128, 32],
        num_classes = config["num_classes"],
        aggregation = 'max',
        apply_dropedge = False,
        apply_bn = True,
        apply_dropout = False,
        # num_heads = 1,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MeshProcessingNetwork(**model_params).to(device) 
    model = model.double()

    print(model)