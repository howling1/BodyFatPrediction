import numpy as np
from tqdm import tqdm
import random
import wandb 

import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

from models.feast_conv import FeatureSteeredConvolution
from models.mesh_processing_net import MeshProcessingNetwork
from models.jk_net import JKNet
from models.dense_gnn import DenseGNN
from models.res_gnn import ResGNN
from datasets.in_memory import IMDataset

REGISTERED_ROOT = "/data1/practical-wise2223/registered_5" # the path of the dir saving the .ply registered data
INMEMORY_ROOT = '/data1/practical-wise2223/registered5_gender_seperation_root' # the root dir path to save all the artifacts ralated of the InMemoryDataset
FEATURES_PATH = "/vol/chameleon/projects/mesh_gnn/basic_features.csv"
TARGET = "age"

def test(model, loader, device, task):
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

def val_epoch(model, valloader, loss_criterion, device, task):
    cumu_loss = 0
    _predictions = torch.tensor([])
    _labels = torch.tensor([])

    # set model to eval, important if your network has e.g. dropout or batchnorm layers
    model.eval()
                
    # forward pass and evaluation for entire validation set
    for val_data in valloader:
        val_data = val_data.to(device)
        
        with torch.no_grad():
            # Get prediction scores
            if task == "classification":
                prediction = F.softmax(model(val_data).detach().cpu(), dim=1)
            elif task == "regression":
                prediction = model(val_data).detach().cpu()
                        
        val_label = val_data.y.detach().cpu()
        #keep track of cumulative loss                                 
        loss = loss_criterion(prediction,val_label)
        cumu_loss += loss.item()
        _predictions = torch.cat((_predictions.double(), prediction.double()))
        _labels = torch.cat((_labels.double(), val_label.double()))

    if task == "classification":
        accuracy = np.mean((torch.argmax(_predictions,1)==torch.argmax(_labels,1)).numpy())
    elif task == "regression":
        accuracy = r2_score(_labels, _predictions)

    wandb.log({"val batch loss": loss.item(), "acc metric":accuracy})
    # set model back to train
    model.train()

    return cumu_loss / len(valloader)

def train_epoch(model, trainloader, optimizer, loss_criterion, device):
    cumu_loss = 0
    for i, data in tqdm(enumerate(trainloader)):  
        data = data.to(device)
        optimizer.zero_grad()
        prediction = model(data)
        label = data.y.to(device)
        loss = loss_criterion(prediction, label)  
        cumu_loss += loss.item()
        loss.to(device)
        loss.backward()
        optimizer.step()
        wandb.log({"training batch loss": loss.item()})

    return cumu_loss / len(trainloader)
     
def train(config=None):
    #set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("current GPU:", torch.cuda.get_device_name(device))
    print("using GPU:", torch.cuda.current_device())
    #---------------------

    #initialize dataloaders
    dataset_female = IMDataset(REGISTERED_ROOT, INMEMORY_ROOT, FEATURES_PATH, TARGET, 0)
    dataset_male = IMDataset(REGISTERED_ROOT, INMEMORY_ROOT, FEATURES_PATH, TARGET, 1)

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
    #----------------------------

    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        train_loader = DataLoader(train_data_all, batch_size = config.batch_size, shuffle = True)
        val_loader = DataLoader(val_data_all, batch_size = config.batch_size, shuffle = True)
        
        # initialize model parameters
        n_class = 1 if config.task == "regression" else 2
        model_params = dict(
            gnn_conv = SAGEConv,
            in_features = 3,
            num_hiddens = config.num_hiddens,
            num_layers = config.num_layers,
            num_skip_layers = config.num_skip_layers,
            encoder_channels = config.encoder_channels,
            decoder_channels = [256, 32],
            num_classes = n_class,
            aggregation = config.aggregation, # mean, max
            apply_dropedge = config.apply_dropedge,
            apply_bn = True,
            apply_dropout = True
        )

        model = ResGNN(**model_params).to(device)
        model = model.double()
        #-----------------------

        #initialize criterion
        if config.task== "regression" :
            loss_criterion = torch.nn.MSELoss()
        elif config.task== "classification": 
            loss_criterion = torch.nn.CrossEntropyLoss()

        loss_criterion.to(device)
        #-----------------------

        #initialize optimizer and set model to train
        optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate, weight_decay=config.weight_decay)

        model.train()

        for epoch in range(config.epochs):
            training_loss = train_epoch(model, train_loader, optimizer, loss_criterion, device)
            val_loss = val_epoch(model, val_loader, loss_criterion, device, config.task)
            wandb.log({"training_loss": training_loss, "val_loss":val_loss, "epoch": epoch})           

def main(): 

    sweep_config = {
        'method': 'random' # matches the parameters randomly, can use 'grid' as well to match each parameter with each other
    } 

    metric = {
        'name' : 'val_loss',
        'goal' : 'minimize'
    }

    sweep_config['metric'] = metric

    parameters_dict = {
        'num_hiddens' : { 'values' : [16, 32, 64]},   
        'num_layers' : { 'values': [2, 4, 6, 8, 10]},
        'num_skip_layers': { 'values': [1, 2]},
        'aggregation': {'values': ['max', 'mean']},   
        'apply_dropedge': {'values': [True, False]},
        'apply_dropout': {'values': [True, False]},
        'encoder_channels': {'values': [[], [64]]},
        'learning_rate': { 'distribution': 'uniform', 'min': 0,  'max': 0.01 }, # need to give a distribution for it to pick the parameter while using 'random' search 
        'weight_decay': {'distribution': 'uniform', 'min': 0.0001,  'max': 0.02 },
        'epochs' : { 'value' : 800}, # set parameter only single value if you don't want it to change during sweep
        'batch_size' : {'values' : [16, 32, 64]},
        'task' : { 'value' : 'regression' } # "regression" or "classification"
    }

    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project="mesh-gnn_sweep") 

    wandb.agent(sweep_id, train, count=5) # count parameter is necessary for random search, it stops after reaching count. grid search stops when all the possibilites finished.
    
if __name__ == "__main__":
    torch.cuda.set_device(0)
    main()
