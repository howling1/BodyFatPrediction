import wandb
import torch
import os
import torch.nn as nn
from helper_functions import get_data_and_labels, get_dataloaders, train
from torchvision import models


def main():
    DATA_ROOT = "/vol/space/projects/ukbb/projects/silhouette/silhouettes/v1"
    FEATURES_ROOT = "/vol/space/projects/ukbb/projects/silhouette/ukb668815_imaging.csv"
    MODEL_ROOT = "./best-models/"
    
    config = {
        "target" : "ASAT",
        "experiment_name" : "densenet-asat",
        "pretrain" : True,
        "freeze" : False,
        # config for training
        "batch_size" : 32,
        "epochs" : 50,
        "decayed_lr": True,
        "base_lr" : 1e-4,
        "clip_norm": 1.0, # set to None if not needed
        "weight_decay": 1e-4,
        "print_every_n" : 20,
        "validate_every_n" : 20,
        # config for densenet
        "fc_dims" : [512, 128, 32],
        "out_dim" : 1, # 1 = regression or >1 = classification
        }
    
    data, targets = get_data_and_labels(DATA_ROOT, FEATURES_ROOT, config['target'], config['out_dim'])
    data = torch.cat([data]*3, dim=1)
    trainloader, valloader, testloader = get_dataloaders(data, targets, config['batch_size'])

    model_path = MODEL_ROOT + config['experiment_name']
    if not os.path.exists(MODEL_ROOT + config['experiment_name']):
        os.makedirs(model_path)
    else:
        raise Exception("Folder already exists. Please delete it. Deletion will potentially delete saved models in this folder.")

    wandb.init(project = "silhouette-prediction", config = config) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.densenet121(weights=config['pretrain'])
    
    num_ftrs = model.classifier.in_features
    fc_dims = [num_ftrs] + config['fc_dims']
    fc_layers = [nn.ReLU()]
    for i in range(len(fc_dims) - 1):
        fc_layers.append(nn.Linear(fc_dims[i], fc_dims[i+1]))
        fc_layers.append(nn.ReLU())
    
    fc_layers.append(nn.Linear(fc_dims[-1], config['out_dim']))

    model.classifier = nn.Sequential(*fc_layers)
    for param in model.features.parameters():
            param.requires_grad = not config['freeze']
    model = model.to(device).double()

    train(model, trainloader, valloader, config, device)
    
if __name__ == "__main__":
    main()