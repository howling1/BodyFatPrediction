import wandb
import torch
import pandas as pd
from models import CNN
import os
from helper_functions import get_data_and_labels, get_dataloaders, train
import numpy as np

def main():
    DATA_ROOT = "/vol/space/projects/ukbb/projects/silhouette/silhouettes/v1"
    IDS_PATH = "/vol/space/projects/ukbb/projects/silhouette/eids_filtered.npy"
    FEATURES_ROOT = "/vol/space/projects/ukbb/projects/silhouette/ukb668815_imaging.csv"
    MODEL_ROOT = "./best-models/"
    SIZE = (363, 392)
    IMG_CHANNEL = 1
    
    config = {
        "target" : "VAT",
        "experiment_name" : "cnn-vat",
        # config for training
        "batch_size" : 8,
        "epochs" : 50,
        "decayed_lr": True,
        "base_lr" : 0.001,
        "clip_norm": 1.0, # set to None if not needed
        "weight_decay": 0.,
        "print_every_n" : 10,
        "validate_every_n" : 10,
        # config for cnn
        "cnn_channels" : [32, 64, 128],
        "kernel_sizes" : [3, 3, 3],
        "paddings" : [1, 1, 1],
        "pool_sizes" : [3, 3, 3],
        "fc_dims" : [512],
        "out_dim" : 1, # 1 = regression or >1 = classification
        "regularization" : "bn"
        }

    model_path = MODEL_ROOT + config['experiment_name']
    if not os.path.exists(MODEL_ROOT + config['experiment_name']):
        os.makedirs(model_path)
    # else:
    #     raise Exception("Folder already exists. Please delete it. Deletion will potentially delete saved models in this folder.")

    wandb.init(project = "silhouette-prediction", config = config) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNN(img_width=SIZE[1], 
                    img_height=SIZE[0], 
                    input_channel=IMG_CHANNEL, 
                    cnn_channels=config['cnn_channels'], 
                    kernel_sizes=config['kernel_sizes'], 
                    paddings=config['paddings'], 
                    pool_sizes=config['pool_sizes'], 
                    fc_dims=config['fc_dims'], 
                    out_dim=config['out_dim'], 
                    regularization=config['regularization'],
                    ).to(device).double() 
    
    ids = np.load(IDS_PATH)
    
    data, targets = get_data_and_labels(DATA_ROOT, ids, FEATURES_ROOT)

    trainloader, valloader, testloader = get_dataloaders(data, targets, config['batch_size'])

    run_n_mock = 200

    train(model, trainloader, valloader, config, device, run_n_mock)
    
if __name__ == "__main__":
    main()