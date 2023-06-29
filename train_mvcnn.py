import wandb
import torch
import pandas as pd
from models import MultiviewCNN, RegressionShrinkageLoss
import os
import numpy as np
from sklearn.metrics import r2_score
import torch.nn as nn
from helper_functions import get_data_and_labels, get_dataloaders, train, mock_get_data

def main():
    DATA_ROOT = "/vol/space/projects/ukbb/projects/silhouette/silhouettes/v1"
    FEATURES_ROOT = "/vol/space/projects/ukbb/projects/silhouette/ukb668815_imaging.csv"
    IDS_PATH = "/vol/space/projects/ukbb/projects/silhouette/eids_filtered.npy"
    MODEL_ROOT = "./best-models/"
    SIZE = (363, 392)
    IMG_CHANNEL = 1
    CORONAL_SIZE = 224
    
    config = {
        "target" : "VAT",
        "experiment_name" : "multiviewcnn-vat",
        # config for training
        "batch_size" : 8,
        "epochs" : 150,
        "decayed_lr": True,
        "base_lr" : 0.0001,
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
        "out_dim" : 2, # 1 = regression or >1 = classification
        "regularization" : "bn"
        }

    model_path = MODEL_ROOT + config['experiment_name']
    if not os.path.exists(MODEL_ROOT + config['experiment_name']):
        os.makedirs(model_path)
    # else:
    #     raise Exception("Folder already exists. Please delete it. Deletion will potentially delete saved models in this folder.")

    wandb.init(project = "silhouette-prediction", config = config) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiviewCNN(img_width=SIZE[1], 
                    img_height=SIZE[0], 
                    coronal_width=CORONAL_SIZE,
                    input_channel=IMG_CHANNEL, 
                    cnn_channels=config['cnn_channels'], 
                    kernel_sizes=config['kernel_sizes'], 
                    paddings=config['paddings'], 
                    pool_sizes=config['pool_sizes'], 
                    fc_dims=config['fc_dims'], 
                    out_dim=config['out_dim'], 
                    regularization=config['regularization'],
                    ).to(device).double() 
    
    # data, targets = get_data_and_labels(DATA_ROOT, FEATURES_ROOT, config['target'], config['out_dim'])
    data, targets = mock_get_data(DATA_ROOT, FEATURES_ROOT)
    ids = np.load(IDS_PATH)
    # data, targets = get_data_and_labels(DATA_ROOT, ids, FEATURES_ROOT)
    # male_data, female_data, male_targets, female_targets = get_data_and_labels(DATA_ROOT, ids, FEATURES_ROOT)
    # data = torch.cat((male_data, female_data), 0)
    # targets = torch.cat((male_targets, female_targets), 0)

    print(data.shape)
    print(targets.shape)
    trainloader, valloader, testloader = get_dataloaders(data, targets, config['batch_size'])

    criterion = nn.MSELoss()
    # criterion = RegressionShrinkageLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['base_lr'], weight_decay=config['weight_decay'])

    wandb.init(project = "silhouette-prediction", config = config, name="mock") 

    model.train()

    train_loss_running = 0.
    training_n = 0

    for epoch in range(config['epochs']):
        for i, (images, targets) in enumerate(trainloader):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            print(outputs.shape)
            print(targets.shape)
            # loss = criterion(outputs[:, 0], targets[:, 0]) + criterion(outputs[:, 1], targets[:, 1])
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_running += loss.item() * targets.shape[0]
            iteration = epoch * len(trainloader) + i
            training_n += targets.shape[0]

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                wandb.log({"training loss": train_loss_running / training_n})
                train_loss_running = 0.
                training_n = 0

            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):
                # set model to eval, important if your network has e.g. dropout or batchnorm layers
                _predictions = torch.tensor([]).to(device)
                _labels = torch.tensor([]).to(device)
                model.eval()
                
                # forward pass and evaluation for entire validation set
                for val_data in valloader:
                    x_val, y_val = val_data[0].to(device), val_data[1].to(device)
                    
                    with torch.no_grad():
                        # Get prediction scores
                        prediction = model(x_val)
                                  
                    # y_val = y_val.detach().cpu()
                    #keep track of loss_total_val                                  
                    _predictions = torch.cat((_predictions.double(), prediction.double()))
                    _labels = torch.cat((_labels.double(), y_val.double()))

                accuracy = r2_score(_labels.detach().cpu(), _predictions.detach().cpu(), multioutput='raw_values')
                loss_val = criterion(_predictions, _labels)
                # loss_val = criterion(_predictions[:, 0], _labels[:, 0]) + criterion(_predictions[:, 1], _labels[:, 1])

                print(f'[{epoch:03d}/{i:05d}] val_loss: {loss_val:.3f}')
                wandb.log({"validation loss": loss_val, "epoch": epoch})
                for i, acc in enumerate(accuracy):
                    wandb.log({"val_acc_" + str(i): acc}) 

                model.train()

    # run_n_mock = 300
    # train(model, trainloader, valloader, config, device, run_n_mock)

    
if __name__ == "__main__":
    main()