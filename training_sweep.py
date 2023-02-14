import numpy as np
from tqdm import tqdm
import wandb 
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from models.feast_conv import FeatureSteeredConvolution
from models.mesh_processing_net import MeshProcessingNetwork
from models.jk_net import JKNet
from models.dense_gnn import DenseGNN
from models.res_gnn import ResGNN
from datasets.in_memory import IMDataset
from helper_methods import load_and_split_dataset

REGISTERED_ROOT = "/data1/practical-wise2223/registered_5" # the path of the dir saving the .ply registered data
INMEMORY_ROOT = '/data1/practical-wise2223/registered5_gender_seperation_root' # the root dir path to save all the artifacts ralated of the InMemoryDataset
FEATURES_PATH = "/vol/chameleon/projects/mesh_gnn/basic_features.csv"
TARGET = "age"

def val_epoch(model, valloader, loss_criterion, device, task):
    """
    Function for validation on the hyperparameter optimization with wandb sweep
    :param model: initialized model
    :param valloader: validation data loader
    :param loss_criterion: loss function
    :param device: torch device
    :param task: type of the prediction task
        ["classification","regression"]
    """
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
    """
    Function for training on the hyperparameter optimization with wandb sweep
    :param model: initialized model
    :param trainloader: train data loader
    :param optimizer: initialized optimizer
    :param loss_criterion: loss function
    :param device: torch device
    """
    cumu_loss = 0
    for _, data in tqdm(enumerate(trainloader)):  
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
    """
    training function to pass to the wandb sweep agent
    config will be explained below
    """
    #set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #----------------------------
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        train_data_all, val_data_all, _, _ = load_and_split_dataset(REGISTERED_ROOT, INMEMORY_ROOT, FEATURES_PATH, TARGET)

        train_loader = DataLoader(train_data_all, batch_size = config.batch_size, shuffle = True)
        val_loader = DataLoader(val_data_all, batch_size = config.batch_size, shuffle = True)

        # initialize model parameters
        n_class = 1 if config.task == "regression" else 2
        model_params = dict(
            gnn_conv = SAGEConv,
            in_features = 3,
            num_hiddens = config.num_hiddens,
            num_layers = config.num_layers,
            encoder_channels = config.encoder_channels,
            decoder_channels = [256, 32],
            num_classes = n_class,
            aggregation = config.aggregation, # mean, max
            apply_dropedge = config.apply_dropedge,
            apply_bn = True,
            apply_dropout = config.apply_dropedge,
            jk_mode = config.jk_mode
        )

        model = JKNet(**model_params).to(device)
        model = model.double()
        #-----------------------
        #initialize criterion
        if config.task== "regression" :
            loss_criterion = torch.nn.L1Loss()
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
    # choose a metric to optimize in the process
    metric = {
        'name' : 'val_loss',
        'goal' : 'minimize'
    }

    sweep_config['metric'] = metric
    # can put any variable in here to see the different combinations
    parameters_dict = {
        'num_hiddens' : { 'values' : [16, 32, 64]},   
        'num_layers' : { 'values': [4, 6, 8, 10, 12, 16]},
        'aggregation': {'values': ['max', 'mean']},   
        'apply_dropedge': {'value': False},
        'apply_dropout': {'values': [True, False]},
        'encoder_channels': {'values': [[], [128]]},
        'learning_rate': { 'distribution': 'uniform', 'min': 0,  'max': 0.01 }, # need to give a distribution for it to pick the parameter while using 'random' search 
        'weight_decay': {'distribution': 'uniform', 'min': 0.0001,  'max': 0.01 },
        'epochs' : { 'value' : 400}, # set parameter only single value if you don't want it to change during sweep
        'batch_size' : {'value' : 32},
        'task' : { 'value' : 'regression' }, # "regression" or "classification"
        'jk_mode': {'values': ['cat', 'max', 'lstm']},
    }
    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="mesh-gnn_sweep") 
    wandb.agent(sweep_id, train, count=20) # count parameter is necessary for random search, it stops after reaching count. grid search stops when all the possibilites finished.
    
if __name__ == "__main__":
    torch.cuda.set_device(0)
    print("using GPU:", torch.cuda.current_device())
    main()
