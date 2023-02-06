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
from sklearn.model_selection import train_test_split

from models.feast_conv import FeatureSteeredConvolution
from models.mesh_processing_net import MeshProcessingNetwork
from models.jk_net import JKNet
from models.dense_gnn import DenseGNN
from models.res_gnn import ResGNN
from datasets.in_memory import IMDataset
from helper_methods import evaluate

def load_and_split_dataset(raw_data_root, dataset_root, basic_features_path, target):
    dataset_female = IMDataset(raw_data_root, dataset_root, basic_features_path, target, 0)
    dataset_male = IMDataset(raw_data_root, dataset_root, basic_features_path, target, 1)

    dev_data_female, test_data_female = train_test_split(dataset_female, test_size=0.1, random_state=42, shuffle=True)
    train_data_female, val_data_female = train_test_split(dev_data_female, test_size=0.33, random_state=43, shuffle=True)
    dev_data_male, test_data_male = train_test_split(dataset_male, test_size=0.1, random_state=42, shuffle=True)
    train_data_male, val_data_male = train_test_split(dev_data_male, test_size=0.33, random_state=43, shuffle=True)

    train_data_all = train_data_male + train_data_female
    val_data_all = val_data_male + val_data_female
    test_data_all = test_data_male + test_data_female
    random.shuffle(train_data_all)
    random.shuffle(val_data_all)
    random.shuffle(test_data_all)

    return train_data_all, val_data_all, test_data_male, test_data_female


def train(model, trainloader, valloader, device, config):
    
    if config["task"]== "regression" :
        loss_criterion = torch.nn.MSELoss()
    elif config["task"]== "classification": 
        loss_criterion = torch.nn.CrossEntropyLoss()

    loss_criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    model.train()

    best_accuracy = float("-inf")
    train_loss_running = 0.

    for epoch in range(config['epochs']):
        for i, data in tqdm(enumerate(trainloader)):  
            data = data.to(device)
            optimizer.zero_grad()
            prediction = model(data)
            label = data.y.to(device)
            loss = loss_criterion(prediction, label)  
            loss.to(device)
            loss.backward()
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
                        if config["task"] == "classification":
                            prediction = F.softmax(model(val_data).detach().cpu(), dim=1)
                        elif config["task"] == "regression":
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
                    torch.save(model, f'runs/{config["experiment_name"]}/model_best.pt')
                    best_accuracy = accuracy

                # set model back to train
                model.train()

def main():    
    REGISTERED_ROOT = "/data1/practical-wise2223/registered_5" # the path of the dir saving the .ply registered data
    INMEMORY_ROOT = '/data1/practical-wise2223/registered5_gender_seperation_root' # the root dir path to save all the artifacts ralated of the InMemoryDataset
    FEATURES_PATH = "/vol/chameleon/projects/mesh_gnn/basic_features.csv"
    TARGET = "weight"

    config = {
        "experiment_name" : "weight_prediction_5k", # there should be a folder named exactly this under the folder runs/
        "batch_size" : 32,
        "epochs" : 500,
        "learning_rate" : 0.001,
        "weight_decay": 0.,
        "task" : "regression", # "regression" or "classification"
        "print_every_n" : 200,
        "validate_every_n" : 200}

    wandb.init(project = "mesh-gnn", config = config)
    
    n_class = 1 if config["task"] == "regression" else 2

# MeshProcressingNet params
    model_params = dict(
        gnn_conv = GATConv,
        in_features = 3,
        encoder_channels = [16],
        conv_channels = [32, 64, 128],
        decoder_channels = [32],
        num_classes = n_class,
        apply_dropedge = True,
        apply_bn = True,
        apply_dropout = True,
        num_heads=4
    )


# ResGNN params
    # model_params = dict(
    #     gnn_conv = SAGEConv,
    #     in_features = 3,
    #     num_hiddens = 32,
    #     num_layers = 5,
    #     num_skip_layers = 1,
    #     encoder_channels = [128],
    #     decoder_channels = [256, 32],
    #     num_classes = n_class,
    #     aggregation = 'max', # mean, max
    #     apply_dropedge = True,
    #     apply_bn = True,
    #     apply_dropout = True
    # )

# DenseGNN params
    # model_params = dict(
    #     gnn_conv = SAGEConv,
    #     in_features = 3,
    #     num_hiddens = 4,
    #     num_layers = 5,
    #     encoder_channels = [],
    #     decoder_channels = [4],
    #     num_classes = n_class,
    #     aggregation = 'mean', # mean, max
    #     apply_dropedge = True,
    #     apply_bn = True,
    #     apply_dropout = True,
    # )


# JKNet params
    # model_params = dict(
    #     gnn_conv = GATConv,
    #     in_features = 3,
    #     num_hiddens = 4,
    #     num_layers = 3,
    #     encoder_channels = [],
    #     decoder_channels = [8],
    #     num_classes = n_class,
    #     jk_mode = 'lstm', # cat, max, lstm
    #     aggregation = 'max', # mean, max
    #     apply_dropedge = True,
    #     apply_bn = True,
    #     apply_dropout = True,
    #     num_heads=4,
    # )

    torch.cuda.set_device(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("current GPU:", torch.cuda.get_device_name(device))
    print("using GPU:", torch.cuda.current_device())

    model = MeshProcessingNetwork(**model_params).to(device) 
    # model = ResGNN(**model_params).to(device)
    # model = DenseGNN(**model_params).to(device)
    # model = JKNet(**model_params).to(device)
    model = model.double()

    param_log = {
        'params':{
            'network': str(model),
            'config': str(config),
            'model_params': str(model_params),
        }
    }

    wandb.log({"table": pd.DataFrame(param_log)})

    train_data_all, val_data_all, test_data_male, test_data_female = load_and_split_dataset(REGISTERED_ROOT, INMEMORY_ROOT, FEATURES_PATH, TARGET)

    train_loader = DataLoader(train_data_all, batch_size = config["batch_size"], shuffle = True)
    val_loader = DataLoader(val_data_all, batch_size = config["batch_size"], shuffle = True)
    test_loader_female = DataLoader(test_data_female, batch_size = config["batch_size"])
    test_loader_male = DataLoader(test_data_male, batch_size = config["batch_size"])

    train(model, train_loader, val_loader, device, config)

    # testing
    # model = MeshProcessingNetwork(**model_params).to(device) 
    # model.load_state_dict(torch.load(f'runs/{config["experiment_name"]}/model_best.ckpt'))
    model = torch.load(f'runs/{config["experiment_name"]}/model_best.pt')
    
    loss_test_female, r2_test_female = evaluate(model, test_loader_female, device, config["task"])
    loss_test_male, r2_test_male = evaluate(model, test_loader_male, device, config["task"])
    ratio_male = len(test_data_male) / (len(test_data_female) + len(test_data_male))
    ratio_female = len(test_data_female) / (len(test_data_female) + len(test_data_male))
    loss_test = loss_test_female * ratio_female + loss_test_male * ratio_male
    r2_test = r2_test_female * ratio_female + r2_test_male * ratio_male

    test_result = {
        'params':{
            'loss_test_female': loss_test_female.item(),
            'r2_test_female': r2_test_female.item(),
            'loss_test_male': loss_test_male.item(),
            'r2_test_male': r2_test_male.item(),
            'loss_test': loss_test.item(),
            'r2_test': r2_test.item()
        }
    }

    wandb.log({"table": pd.DataFrame(test_result)})


if __name__ == "__main__":
    main()
