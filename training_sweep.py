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
from models.shrinkage_loss import RegressionShrinkageLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_

REGISTERED_ROOT = "/vol/space/projects/ukbb/projects/silhouette/registered_5" # the path of the dir saving the .ply registered data
INMEMORY_ROOT = '/vol/space/projects/ukbb/projects/silhouette/imdataset/registered5_vat' # the root dir path to save all the artifacts ralated of the InMemoryDataset
FEATURES_PATH = "/vol/space/projects/ukbb/projects/silhouette/ukb668815_imaging.csv"
IDS_PATH = "/vol/space/projects/ukbb/projects/silhouette/eids_filtered.npy"
TARGET = "vat"

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

    wandb.log({"val_epoch_loss": cumu_loss / len(valloader), "val_epoch_acc": accuracy})
    # set model back to train
    model.train()

    return accuracy

def train_epoch(model, trainloader, optimizer, scheduler, loss_criterion, device, config):
    """
    Function for training on the hyperparameter optimization with wandb sweep
    :param model: initialized model
    :param trainloader: train data loader
    :param optimizer: initialized optimizer
    :param scheduler: Cosine learning rate scheduler. None if not needed
    :param loss_criterion: loss function
    :param device: torch device
    :param config: training config
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

        if config['clip_norm'] is not None:
            clip_grad_norm_(model.parameters(), config['clip_norm'])

        optimizer.step()
        if scheduler is not None:
                scheduler.step()
        wandb.log({"training batch loss": loss.item()})

    wandb.log({"train_epoch_loss": cumu_loss / len(trainloader)})
     
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

        train_data_all, val_data_all, test_data_all = load_and_split_dataset(REGISTERED_ROOT, INMEMORY_ROOT, FEATURES_PATH, IDS_PATH, TARGET)

        train_loader = DataLoader(train_data_all, batch_size = config.batch_size, shuffle = True)
        val_loader = DataLoader(val_data_all, batch_size = config.batch_size, shuffle = True)

        # initialize model parameters
        n_class = 1 if config.task == "regression" else 2
        model_params = dict(
            gnn_conv = GATConv,
            in_features = 3,
            conv_channels = config.conv_channels,
            encoder_channels = [],
            decoder_channels = [512, 128, 32],
            num_classes = n_class,
            aggregation = config.aggregation, # mean, max
            num_heads = config.num_heads,
            apply_dropedge = False,
            apply_bn = True,
            apply_dropout = False,
        )

        model = MeshProcessingNetwork(**model_params).to(device)
        model = model.double()
        #-----------------------
        #initialize criterion
        if config.task== "regression" :
            if config['loss'] == "Shrinkage":
                loss_criterion = RegressionShrinkageLoss()
            elif config['loss'] == "MSE":
                loss_criterion = torch.nn.MSELoss()
            elif config['loss'] == "MAE":
                loss_criterion = torch.nn.L1Loss()
        elif config.task== "classification": 
            loss_criterion = torch.nn.CrossEntropyLoss()

        loss_criterion.to(device)
        #-----------------------
        #initialize optimizer and set model to train
        optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate, weight_decay=config.weight_decay)

        scheduler = None
        if config['decayed_lr']:
            scheduler = CosineAnnealingLR(optimizer, config['epochs'])

        model.train()

        best_acc = best_accuracy = float("-inf")

        for epoch in range(config.epochs):
            train_epoch(model, train_loader, optimizer, scheduler, loss_criterion, device, config)
            val_acc = val_epoch(model, val_loader, loss_criterion, device, config.task)
            if val_acc > best_acc:
                best_acc = val_acc
                wandb.log({"best_val_acc": best_acc}) 
            wandb.log({"epoch": epoch})           

def main(): 
    sweep_config = {
        'method': 'grid' # matches the parameters randomly, can use 'grid' as well to match each parameter with each other
    } 
    # choose a metric to optimize in the process
    metric = {
        'name' : 'best_val_acc',
        'goal' : 'maximize'
    }

    sweep_config['metric'] = metric
    # can put any variable in here to see the different combinations
    parameters_dict = {
        'conv_channels' : { 'value' : [32, 64, 128]},
        #'num_hiddens' : { 'values' : [32, 64]},  # for jknet
        #'num_layers' : { 'values': [2 ,3, 4]}, # for jknet
        'aggregation': {'value': 'max'},   
        'learning_rate': {'value' : 1e-3}, # need to give a distribution for it to pick the parameter while using 'random' search 
        'weight_decay': {'value' : 0},
        'epochs' : { 'value' : 50}, # set parameter only single value if you don't want it to change during sweep
        "decayed_lr": {'value' : True},
        "clip_norm": {'value' : 1},
        "num_heads": {'values': [1, 2, 4, 8]},
        'batch_size' : {'value' : 32},
        'loss': {'value': "Shrinkage"},
        'task' : { 'value' : 'regression' }, # "regression" or "classification"
        # "print_every_n" : {'value' : 20},
        # "validate_every_n" : {'value' : 20}
        #'jk_mode': {'values': ['cat', 'max', 'lstm']},
    }
    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="mesh-gnn_sweep") 
    wandb.agent(sweep_id, train, count=96) # count parameter is necessary for random search, it stops after reaching count. grid search stops when all the possibilites finished.
    
if __name__ == "__main__":
    # torch.cuda.set_device(0)
    print("using GPU:", torch.cuda.current_device())
    main()
