from pathlib import Path
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from model import FeatureSteeredConvolution,  MeshProcessingNetwork
from datasets.in_memory import IMDataset
import wandb

TARGET = "bmi"
REGISTERED_ROOT = Path("D:/ADLM_Data/registered_25/") # the path of the directory that includes the .ply registered data
INMEMORY_ROOT = 'D:/ADLM_Data/registered25_InMemoryDataset_root' # the root directory to save all the artifacts related of the InMemoryDataset
PROCESSED_PATH = 'D:/ADLM_Data/registered25_InMemoryDataset_root/'+ TARGET + '_dataset.pt' # the path of the InMemoryDataset file to be created

raw_file_paths = [os.path.join(str(REGISTERED_ROOT), file).replace('\\', '/') for file in os.listdir(str(REGISTERED_ROOT))]
basic_features_path = "D:/ADLM_Data/basic_features.csv"
basic_features = pd.read_csv(basic_features_path)

def train(model, trainloader, valloader, device, config):
    
    if config["task"]== "regression" :
        loss_criterion = torch.nn.MSELoss()
    elif config["task"]== "classification": 
        loss_criterion = torch.nn.CrossEntropyLoss()

    loss_criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    model.train()

    best_accuracy = 0.
    loss_all = 0
    train_loss_running = 0.

    for epoch in range(config['epochs']):
        for i, data in tqdm(enumerate(trainloader)):  
            data = data.to(device)
            optimizer.zero_grad()
            prediction = model(data)
            label = data.y.to(device)
            #print(prediction.shape)
            #print(label.shape)
            loss = loss_criterion(prediction, label)  
            loss.to(device)
            loss.backward()
            loss_all += data.num_graphs * float(loss)
            optimizer.step()
            
            # loss logging
            train_loss_running += loss.item()
            iteration = epoch * len(trainloader) + i

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                print(f'[{epoch:03d}/{i:05d}] train_loss: {train_loss_running / config["print_every_n"]:.3f}')
                wandb.log({"training loss": train_loss_running / config["print_every_n"]})
                train_loss_running = 0.
                
            # validation evaluation and logging
            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):

                loss_total_val = 0
                _predictions = torch.tensor([])
                _labels = torch.tensor([])

                # set model to eval, important if your network has e.g. dropout or batchnorm layers
                model.eval()
                
                # forward pass and evaluation for entire validation set
                for val_data in valloader:
                    val_data = val_data.to(device)
                    
                    with torch.no_grad():
                        # Get prediction scores
                        prediction = model(val_data).detach().cpu()
                                  
                    val_label = val_data.y.detach().cpu()
                    #keep track of loss_total_val                                  
                    loss_total_val += loss_criterion(prediction,val_label).item()
                    _predictions = torch.cat((_predictions.double(), prediction.double()))
                    _labels = torch.cat((_labels.double(), val_label.double()))

                if config["task"] == "classification":
                    accuracy = np.mean((torch.argmax(_predictions,1)==torch.argmax(_labels,1)).numpy())
                elif config["task"] == "regression":
                    accuracy = r2_score(_labels, _predictions)

                print(f'[{epoch:03d}/{i:05d}] val_loss: {loss_total_val / len(valloader):.3f}, val_accuracy: {accuracy:.3f}')
                wandb.log({"validation loss": loss_total_val / len(valloader), "validation accuracy": accuracy })
                if accuracy > best_accuracy:
                    torch.save(model.state_dict(), f'runs/{config["experiment_name"]}/model_best.ckpt')
                    best_accuracy = accuracy

                # set model back to train
                model.train()

def main():

    #dataset = IMDataset(REGISTERED_ROOT, INMEMORY_ROOT, basic_features_path, TARGET)
    
    dataset_female = IMDataset(REGISTERED_ROOT, INMEMORY_ROOT, basic_features_path, TARGET, 0)
    dataset_male = IMDataset(REGISTERED_ROOT, INMEMORY_ROOT, basic_features_path, TARGET, 1)

    dev_data_female, test_data_female = train_test_split(dataset_female, test_size=0.2, random_state=42, shuffle=True)
    train_data_female, val_data_female = train_test_split(dev_data_female, test_size=0.25, random_state=43, shuffle=True)
    dev_data_male, test_data_male = train_test_split(dataset_male, test_size=0.2, random_state=42, shuffle=True)
    train_data_male, val_data_male = train_test_split(dev_data_male, test_size=0.25, random_state=43, shuffle=True)

    train_data_all = train_data_male + train_data_female
    val_data_all = val_data_male + val_data_female
    test_data_all = test_data_male + test_data_female
    random.shuffle(train_data_all)
    random.shuffle(val_data_all)
    random.shuffle(test_data_all)

    """ dev_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)
    train_data, val_data = train_test_split(dev_data, test_size=0.2, random_state=43, shuffle=True)

    print("size of train_data:",len(train_data))
    print("size of val_data:",len(val_data))
    print("size of test_data:", len(test_data))
 """

    config = {
        "experiment_name" : "sex_prediction_25k", # there should be a folder named exactly this under the folder runs/
        "batch_size" : 4,
        "epochs" : 10,
        "learning_rate" : 0.001,
        "task" : "regression", # "regression" or "classification"
        "print_every_n" : 200,
        "validate_every_n" : 200}

    wandb.init(project = "mesh-gnn", config = config)

    n_class = 1 if config["task"] == "regression" else 2

    model_params = dict(
    GNN_conv = FeatureSteeredConvolution, # GCN
    in_features = 3,
    encoder_channels = [16],
    conv_channels = [32, 64, 128, 64],
    decoder_channels = [32],
    num_classes = n_class,
    apply_batch_norm = True,
    gf_encoder_params = dict(
        num_heads = 1,
        ensure_trans_invar = True,
        bias = True,
        with_self_loops = True
        )
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("current GPU:", torch.cuda.get_device_name(device))

    model = MeshProcessingNetwork(**model_params).to(device)
    model = model.double()

    train_loader = DataLoader(train_data_all, batch_size = config["batch_size"], shuffle = True)
    val_loader = DataLoader(val_data_all, batch_size = config["batch_size"], shuffle = True)
    #test_loader = DataLoader(test_data, batch_size = config["batch_size"])

    train(model, train_loader, val_loader, device, config)

if __name__ == "__main__":
    main()
