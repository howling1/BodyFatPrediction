import wandb
import torch
import os
import torch.nn as nn
from helper_functions import get_data_and_labels, get_dataloaders, train, evaluate
from torchvision import models
import numpy as np
import pandas as pd
from models import RegressionShrinkageLoss, CNN
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import TensorDataset, DataLoader
from codecarbon import EmissionsTracker
import time

def main():
    # DATA_ROOT = "/vol/space/projects/ukbb/projects/silhouette/silhouettes/v1"
    # IDS_PATH = "/vol/space/projects/ukbb/projects/silhouette/eids_filtered.npy"
    # FEATURES_ROOT = "/vol/space/projects/ukbb/projects/silhouette/ukb668815_imaging.csv"
    TENSOR_PATH = "/vol/space/projects/ukbb/projects/silhouette/silhouettes/sih_male_female"
    MODEL_ROOT = "./best-models/"
    
    config = {
        "experiment_name" : "densenet_cv",
        "pretrain" : True,
        "freeze" : False,
        # config for training
        "batch_size" : 32,
        "epochs" : 100,
        "decayed_lr": True,
        "base_lr" : 1e-4,
        "clip_norm": 1.0, # set to None if not needed
        "weight_decay": 1e-4,
        "print_every_n" : 100,
        "validate_every_n" : 100,
        # config for densenet
        "fc_dims" : [512,128,32],
        "out_dim" : 2,
        "early_stopping_patience" : 30
        }
    
    # ids = np.load(IDS_PATH)
    # data_male, data_female, targets_male, targets_female = get_data_and_labels(DATA_ROOT, ids, FEATURES_ROOT)
    data_male = torch.load(TENSOR_PATH + "/male_data.pt")
    data_female = torch.load(TENSOR_PATH + "/female_data.pt")
    targets_male = torch.load(TENSOR_PATH + "/male_targets.pt")
    targets_female = torch.load(TENSOR_PATH + "/female_targets.pt")

    data_male = torch.cat([data_male]*3, dim=1)
    data_female = torch.cat([data_female]*3, dim=1)

    model_path = MODEL_ROOT + config['experiment_name']
    if not os.path.exists(MODEL_ROOT + config['experiment_name']):
        os.makedirs(model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dev_female_x, test_female_x, dev_female_y, test_female_y = train_test_split(data_female, targets_female, test_size=1/6, random_state=42)
    dev_male_x, test_male_x, dev_male_y, test_male_y = train_test_split(data_male, targets_male, test_size=1/6, random_state=42)
    
    test_female_dataset = TensorDataset(test_female_x, test_female_y)
    test_male_dataset = TensorDataset(test_male_x, test_male_y)
    test_dataset = TensorDataset(torch.cat([test_male_x, test_female_x], dim=0), torch.cat([test_male_y, test_female_y], dim=0))

    female_testloader = DataLoader(test_female_dataset, batch_size = config["batch_size"])
    male_testloader = DataLoader(test_male_dataset, batch_size = config["batch_size"])
    testloader = DataLoader(test_dataset, batch_size = config["batch_size"])

    k = 5
    kf = KFold(n_splits=k, shuffle=False)
    run_n = 1

    for (train_index_female, val_index_female), (train_index_male, val_index_male) in zip(kf.split(dev_female_x), kf.split(dev_male_x)):
        train_female_x = dev_female_x[train_index_female]
        val_female_x = dev_female_x[val_index_female]
        train_male_x = dev_male_x[train_index_male]
        val_male_x = dev_male_x[val_index_male]

        train_female_y = dev_female_y[train_index_female]
        val_female_y = dev_female_y[val_index_female]
        train_male_y = dev_male_y[train_index_male]
        val_male_y = dev_male_y[val_index_male]

        train_x = torch.cat([train_female_x, train_male_x], dim=0)
        val_x = torch.cat([val_female_x, val_male_x], dim=0)
        train_y = torch.cat([train_female_y, train_male_y], dim=0)
        val_y = torch.cat([val_female_y, val_male_y], dim=0)

        train_dataset = TensorDataset(train_x, train_y)
        val_dataset = TensorDataset(val_x, val_y)
        val_female_dataset = TensorDataset(val_female_x, val_female_y)
        val_male_dataset = TensorDataset(val_male_x, val_male_y)

        trainloader = DataLoader(train_dataset, batch_size = config["batch_size"], shuffle = True)
        valloader = DataLoader(val_dataset, batch_size = config["batch_size"])
        female_valloader = DataLoader(val_female_dataset, batch_size = config["batch_size"])
        male_valloader = DataLoader(val_male_dataset, batch_size = config["batch_size"])

        model = create_model(config, device)
        # model = CNN(img_width=392, 
        #         img_height=363, 
        #         input_channel=3, 
        #         cnn_channels=[32, 64, 128], 
        #         kernel_sizes=[3, 3, 3], 
        #         paddings=[1, 1, 1], 
        #         pool_sizes=[3, 3, 3], 
        #         fc_dims=[512,128,32], 
        #         out_dim=config["out_dim"], 
        #         regularization="bn",
        #         ).to(device).float() 

        tracker = EmissionsTracker()
        tracker.start()
        start = time.time()
        train(model, trainloader, valloader, config, device, run_n)
        end = time.time()
        duration = end-start
        emissions = tracker.stop()

        model = torch.load(f'./best-models/{config["experiment_name"]}/run{run_n}_model_best.pt')

        acc_val_female = evaluate(model, female_valloader, device)
        acc_val_male = evaluate(model, male_valloader, device)
        acc_val = evaluate(model, valloader, device)
        
        acc_test_female = evaluate(model, female_testloader, device)
        acc_test_male = evaluate(model, male_testloader, device)
        acc_test = evaluate(model, testloader, device)

        test_result = {
            'params':{
                'duration': "duration:" + str(duration),
                'emissions': "emissions:" + str(emissions),
                'acc_val_female': 'R2_val_female: ' + str(np.array(acc_val_female).tolist()),
                'acc_val_male': 'R2_val_male: ' + str(np.array(acc_val_male).tolist()),
                'acc_val': 'R2_val: ' + str(np.array(acc_val).tolist()),
                'acc_test_female': 'R2_test_female: ' + str(np.array(acc_test_female).tolist()),
                'acc_test_male': 'R2_test_male: ' + str(np.array(acc_test_male).tolist()),
                'acc_test': 'R2_test: ' + str(np.array(acc_test).tolist())
            }
        }

        wandb.log({"table": pd.DataFrame(test_result)})
        run_n += 1

def create_model(config, device):
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
    model = model.to(device).float()

    return model
    

if __name__ == "__main__":
    main()