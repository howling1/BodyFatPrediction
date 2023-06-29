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
from models.mesh_processing_net import MeshProcessingNetwork
from helper_methods import evaluate, load_and_split_dataset
from models.shrinkage_loss import RegressionShrinkageLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from datasets.in_memory import IMDataset
from sklearn.model_selection import train_test_split, KFold
from codecarbon import EmissionsTracker
import time


def train_cv(model_params, female_dataset, male_dataset, device, config):
    """
    Function for simple training
    :param model_params: params for model
    :param trainloader: train data loader
    :param valloader: validation data loade
    :param optimizer: initialized optimizer
    :param loss_criterion: loss function
    :param device: torch device
    :param config:
                "experiment_name" : model name which will be saved as the best validation results, 
                                    there should be a matching folder under runs/
                "batch_size" : batch size
                "epochs" : epoch number
                "cyclical_lr": [True,False] # True for cyclical learning rate 
                "base_lr" : learning rate
                "max_lr":  only applicable when cyclical_lr is True
                "weight_decay": weight decay ,
                "task" :  "regression" or "classification"
                "print_every_n" : frequency of logging the batch loss,
                "validate_every_n" : frequency for validation
    """

    female_list = list(female_dataset)
    male_list = list(male_dataset)

    dev_female, test_female = train_test_split(female_list, test_size=1/6, random_state=42)
    dev_male, test_male = train_test_split(male_list, test_size=1/6, random_state=42)
    female_testloader = DataLoader(test_female, batch_size = config["batch_size"])
    male_testloader = DataLoader(test_male, batch_size = config["batch_size"])
    testloader = DataLoader(test_male + test_female, batch_size = config["batch_size"])

    k = 5
    kf = KFold(n_splits=k, shuffle=False)
    run_n = 1

    for (train_index_female, val_index_female), (train_index_male, val_index_male) in zip(kf.split(dev_female), kf.split(dev_male)):
        # if run_n != 5:
        #     run_n += 1
        #     continue

        train_female = [dev_female[i] for i in train_index_female]
        val_female = [dev_female[i] for i in val_index_female]
        train_male = [dev_male[i] for i in train_index_male]
        val_male = [dev_male[i] for i in val_index_male]

        train_data = train_female + train_male
        val_data = val_female + val_male

        trainloader = DataLoader(train_data, batch_size = config["batch_size"], shuffle = True)
        valloader = DataLoader(val_data, batch_size = config["batch_size"])
        female_valloader = DataLoader(val_female, batch_size = config["batch_size"])
        male_valloader = DataLoader(val_male, batch_size = config["batch_size"])

        model = MeshProcessingNetwork(**model_params).float().to(device)
        tracker = EmissionsTracker()
        tracker.start()
        start = time.time()
        train(model, trainloader, valloader, config, device, run_n)
        end = time.time()
        emissions = tracker.stop()
        duration = end-start

        # testing
        acc_name = 'R2' if config['task'] == "regression" else 'Accuracy'
        model = torch.load(f'runs/{config["experiment_name"]}/run{run_n}_model_best.pt')

        _, acc_val_female = evaluate(model, female_valloader, device, config)
        _, acc_val_male = evaluate(model, male_valloader, device, config)
        _, acc_val = evaluate(model, valloader, device, config)
        
        _, acc_test_female = evaluate(model, female_testloader, device, config)
        _, acc_test_male = evaluate(model, male_testloader, device, config)
        _, acc_test = evaluate(model, testloader, device, config)

        test_result = {
            'params':{
                'emissions': "emission:" + str(emissions),
                'duration': "duration:" + str(duration),
                'acc_val_female': acc_name + '_val_female: ' + str(np.array(acc_val_female).tolist()),
                'acc_val_male': acc_name + '_val_male: ' + str(np.array(acc_val_male).tolist()),
                'acc_val': acc_name + '_val: ' + str(np.array(acc_val).tolist()),
                'acc_test_female': acc_name + '_test_female: ' + str(np.array(acc_test_female).tolist()),
                'acc_test_male': acc_name + '_test_male: ' + str(np.array(acc_test_male).tolist()),
                'acc_test': acc_name + '_test: ' + str(np.array(acc_test).tolist())
            }
        }

        wandb.log({"table": pd.DataFrame(test_result)})
        run_n += 1

def train(model, trainloader, valloader, config, device, run_n):
    if config["task"]== "regression" :
        loss_criterion = RegressionShrinkageLoss()
    elif config["task"]== "classification": 
        loss_criterion = torch.nn.CrossEntropyLoss()

    loss_criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['base_lr'], weight_decay=config['weight_decay'])
    scheduler = None
    if config['decayed_lr']:
            scheduler = CosineAnnealingLR(optimizer, config['epochs'])

    model.train()

    # best_accuracy = float("-inf")
    patience_counter = 0
    best_loss = float("+inf")
    train_loss_running = 0.
    training_n = 0

    wandb.init(project = "mesh-gnn", config = config, name=str(run_n), reinit=True) 

    for epoch in range(config['epochs']):
        for i, data in tqdm(enumerate(trainloader)):  
            data = data.to(device)
            optimizer.zero_grad()
            prediction = model(data)
            label = data.y.reshape((-1, config["num_classes"]))
            loss = loss_criterion(prediction, label)  
            loss.to(device)
            loss.backward()

            if config['clip_norm'] is not None:
                clip_grad_norm_(model.parameters(), config['clip_norm'])

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            # loss logging
            train_loss_running += loss.item() * label.shape[0]
            iteration = epoch * len(trainloader) + i
            training_n += label.shape[0]

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                print(f'[{epoch:03d}/{i:05d}] train_loss: {train_loss_running / training_n:.3f}')
                wandb.log({"training loss": train_loss_running / training_n})
                train_loss_running = 0.
                training_n = 0
                
            # validation evaluation and logging
            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):
                _predictions = torch.tensor([]).to(device)
                _labels = torch.tensor([]).to(device)

                # set model to eval, important if your network has e.g. dropout or batchnorm layers
                model.eval()
                
                # forward pass and evaluation for entire validation set
                for val_data in valloader:
                    val_data = val_data.to(device)
                    
                    with torch.no_grad():
                        # Get prediction scores
                        if config["task"] == "classification":
                            prediction = F.softmax(model(val_data), dim=1)
                        elif config["task"] == "regression":
                            prediction = model(val_data)
                                
                    val_label = val_data.y.reshape((-1, config["num_classes"]))
                    #keep track of loss_total_val                              
                    _predictions = torch.cat((_predictions.float(), prediction.float()))
                    _labels = torch.cat((_labels.float(), val_label.float()))

                if config["task"] == "classification":
                    accuracy = np.mean((torch.argmax(_predictions,1)==torch.argmax(_labels,1)).numpy())
                elif config["task"] == "regression":
                    accuracy = r2_score(_labels.detach().cpu(), _predictions.detach().cpu(), multioutput='raw_values')

                loss_val = loss_criterion(_predictions, _labels)

                print(f'[{epoch:03d}/{i:05d}] val_loss: {loss_val:.3f}')
                wandb.log({"validation loss": loss_val, "epoch": epoch})
                for i, acc in enumerate(accuracy):
                        wandb.log({"val_acc_" + str(i): acc}) 

        
                # if accuracy > best_accuracy:
                if loss_val < best_loss:
                    patience_counter = 0

                    # best_accuracy = accuracy
                    best_loss = loss_val

                    print("new best loss:", best_loss)
                    wandb.log({"best_loss": best_loss}) 

                    # wandb.log({"best_val_acc": best_accuracy}) 
                    for i, best_acc in enumerate(accuracy):
                        wandb.log({"best_val_acc_" + str(i): best_acc}) 

                    torch.save(model.state_dict(), f'runs/{config["experiment_name"]}/run{run_n}_model_best.ckpt')
                    torch.save(model, f'runs/{config["experiment_name"]}/run{run_n}_model_best.pt')
                    
                    

                # else:
                #     patience_counter += 1
                #     print("patience_counter:", patience_counter)
                #     if patience_counter >= config["early_stopping_patience"]:
                #         print(f"Early stopping at epoch {epoch+1}")
                #         return

                # set model back to train
                model.train()

def main():    
    REGISTERED_ROOT = "/vol/space/projects/ukbb/projects/silhouette/registered_10k_faces" # the path of the dir saving the .ply registered data
    INMEMORY_ROOT = '/vol/space/projects/ukbb/projects/silhouette/imdataset/registered_10k_faces_imdataset' # the root dir path to save all the artifacts ralated of the InMemoryDataset
    FEATURES_PATH = "/vol/space/projects/ukbb/projects/silhouette/ukb668815_imaging.csv"   
    IDS_PATH = "/vol/space/projects/ukbb/projects/silhouette/eids_filtered.npy"
    TARGET = "all"

    config = {
        "experiment_name" : "cv_all_sage_10k_faces", # there should be a folder named exactly this under the folder runs/
        "batch_size" : 32,
        "epochs" : 150,
        "base_lr" : 0.001,
        "decayed_lr": True,
        "weight_decay": 0.,
        "clip_norm": 1,
        "num_classes": 2,
        "task" : "regression", # "regression" or "classification"
        "print_every_n" : 1000,
        "validate_every_n" : 1000,
        "early_stopping_patience" : 30,
        }
 
    # n_class = 1 if config["task"] == "regression" else 2

# template for  MeshProcressingNet params
    model_params = dict(
        gnn_conv = SAGEConv,
        in_features = 3,
        encoder_channels = [],
        conv_channels = [32, 64, 128],
        decoder_channels = [512, 128, 32],
        num_classes = config["num_classes"],
        aggregation = 'max',
        apply_dropedge = False,
        apply_bn = True,
        apply_dropout = False,
        # num_heads = 1,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_female = IMDataset(REGISTERED_ROOT, INMEMORY_ROOT, FEATURES_PATH, IDS_PATH, TARGET, 0)
    dataset_male = IMDataset(REGISTERED_ROOT, INMEMORY_ROOT, FEATURES_PATH, IDS_PATH, TARGET, 1)

    dataset_female.data.x = dataset_female.data.x.float()
    dataset_female.data.y = dataset_female.data.y.float()
    # dataset_female.data.edge_index = dataset_female.data.edge_index.int()
    dataset_male.data.x = dataset_male.data.x.float()
    dataset_male.data.y = dataset_male.data.y.float()
    # dataset_male.data.edge_index = dataset_male.data.edge_index.int()

    train_cv(model_params, dataset_female, dataset_male, device, config)

if __name__ == "__main__":
    torch.cuda.set_device(0)
    print("using GPU:", torch.cuda.current_device())
    main()


