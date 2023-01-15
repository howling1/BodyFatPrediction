from pathlib import Path
import open3d as o3d
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import math
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from model import FeatureSteeredConvolution,  MeshProcessingNetwork
from datasets.in_memory import IMDataset
import wandb
wandb.init(project="mesh-gnn")

REGISTERED_ROOT = Path("D:/ADLM_Data/registered_5/") # the path of the directory that includes the .ply registered data
INMEMORY_ROOT = 'D:/ADLM_Data/registered_5/registered5_InMemoryDataset_root' # the root directory to save all the artifacts related of the InMemoryDataset
PROCESSED_PATH = 'D:/ADLM_Data/registered_5/registered5_InMemoryDataset_root/height_dataset.pt' # the path of the InMemoryDataset file to be created

raw_file_paths = [os.path.join(str(REGISTERED_ROOT), file).replace('\\', '/') for file in os.listdir(str(REGISTERED_ROOT))]
basic_features_path = "D:/ADLM_Data/basic_features.csv"
basic_features = pd.read_csv(basic_features_path)

TARGET = "height"

def train(device, model, optimizer, crit, loader, dataset):
    model.train()
    loss_all = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * float(loss)
        optimizer.step()

    loss_mean = loss_all / len(dataset)
        
    return loss_mean

# return accuracy
def classification_evaluate(device, model, loader):
    model.eval()
 
    predictions = torch.tensor([])
    labels = torch.tensor([])
 
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data).detach().cpu()
            label = data.y.detach().cpu()
            predictions = torch.cat((predictions, pred))
            labels = torch.cat((labels, label))

        return np.mean((torch.argmax(predictions,1)==torch.argmax(labels,1)).numpy())

def regression_evaluate(device, model, loader, crit):
    model.eval()
 
    predictions = torch.tensor([])
    targets = torch.tensor([])
 
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data).detach().cpu()
            target = data.y.detach().cpu()
            predictions = torch.cat((predictions, pred))
            targets = torch.cat((targets, target))

        loss = crit(predictions, targets)
        r2 = r2_score(targets, predictions)

    return loss, r2

def main():
    dataset = IMDataset(REGISTERED_ROOT, INMEMORY_ROOT, basic_features_path, TARGET)

    dev_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)
    train_data, val_data = train_test_split(dev_data, test_size=0.2, random_state=43, shuffle=True)

    print("size of train_data:",len(train_data))
    print("size of val_data:",len(val_data))
    print("size of test_data:", len(test_data))

    batch_size = 16
    epochs = 10
    lr=0.001
    _regression=1

    model_params = dict(
    GNN_conv = FeatureSteeredConvolution,
    in_features=3,
    encoder_channels=[16],
    conv_channels=[32, 64, 128, 64],
    decoder_channels=[32],
    num_classes=1,
    apply_batch_norm=True,
    gf_encoder_params = dict(
        num_heads=2,
        ensure_trans_invar=True,
        bias=True,
        with_self_loops=True
        )
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("current GPU:", torch.cuda.get_device_name(device))
    model_FeatureSteered = MeshProcessingNetwork(**model_params).to(device)
    #model_FeatureSteered = MeshClassif(**model_params).to(device)
    model_FeatureSteered = model_FeatureSteered.double()
    optimizer = torch.optim.Adam(model_FeatureSteered.parameters(), lr)
    
    if _regression==1:
        crit = torch.nn.MSELoss()
    elif _regression==0: 
        crit = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    for epoch in range(epochs):
        train_loss = train(device=device, model=model_FeatureSteered, optimizer=optimizer, crit=crit, loader=train_loader, dataset=train_data)
        
        _, train_acc = regression_evaluate(device, model_FeatureSteered, train_loader, crit )
        val_loss, val_acc = regression_evaluate(device, model_FeatureSteered, val_loader, crit) 
        
        print('Epoch: {:03d}, Train Loss: {:.5f}, Train Accuracy: {:.5f}, Val Accuracy: {:.5f}'.
            format(epoch, train_loss, train_acc, val_acc))

        wandb.log({"training accuracy": train_acc, "training loss": train_loss, "validation loss": val_loss,"validation acc": val_acc})

    test_loss, test_acc = regression_evaluate(device, model_FeatureSteered, test_loader, crit)
    print('Test Loss: {:.5f}, Test Accuracy: {:.5f}'.format(test_loss, test_acc))
    wandb.finish()

if __name__ == "__main__":
    main()
