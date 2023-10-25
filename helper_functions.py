import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
import pandas as pd
import torch
import wandb
import torch.nn.functional as F
from PIL import Image as im
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from models import RegressionShrinkageLoss
import torch.nn as nn

def process(DATA_ROOT, LIMIT, TARGET_ROOT, SIZE, EXTENSION):
    """
    create coronal and sagittal ssilhouettes from segamentation and concatenate them together for the whole dataset
    :param DATA_ROOT: path for the segmentation data 
    :param TARGET_ROOT: path where the generated silhouette imgs will be saved
    :param EXTENSION: extension of the files that will be saved
    :param LIMIT: how many files to process
    :param SIZE: the size of a generated silhouette img
    """
    COUNT = 0

    for file in tqdm(os.listdir(DATA_ROOT)):
        _path = str(os.path.join(str(DATA_ROOT), file).replace('\\', '/')) +'/body_mask.nii.gz'
        _id = _path[_path[:_path.rfind("/")].rfind("/")+1:_path.rfind("/",0,)]
        
        if (LIMIT > COUNT) :
            if(os.path.exists(_path)):
            
                body_segment = nib.load(_path)
                body_segment_data = body_segment.get_fdata()

                slh_coronal = create_silhouette(body_segment_data, 1)
                slh_sagittal = create_silhouette(body_segment_data, 0)
                concatenated_sil = cat_silhouette(slh_coronal, slh_sagittal, SIZE)

                _target_path =  TARGET_ROOT + "/" + _id + EXTENSION

                concatenated_sil.save(_target_path)

                COUNT += 1
            else:
                continue
        else:
            break

def create_silhouette(body_segment_data: np.ndarray, direction: int) -> np.ndarray:
    """create silhouette from segmentation according to the given direction"""
    tmp = body_segment_data.mean(axis=direction)
    shape = tmp.shape
    oneD_data = np.ravel(tmp)
    slh = []

    for i in range(oneD_data.shape[0]):
        if oneD_data[i] != 0.:
            slh.append(255.)
        else:
            slh.append(0.)   
            
    slh = np.array(slh).reshape(shape)

    if direction == 0: # transpose sagittal slh
        slh = np.asarray(im.fromarray(slh).transpose(im.FLIP_TOP_BOTTOM))

    return slh

def cat_silhouette(slh_coronal: np.ndarray, slh_sagittal: np.ndarray, size: tuple) -> im.Image:
    """concatenate coronal and sagittal silhouettes"""
    concatenated_slh = np.concatenate((slh_coronal, slh_sagittal), 0)
    slh_image = im.fromarray(concatenated_slh)

    return slh_image.resize((max(size),max(size))).rotate(90).resize(size).convert('L') 

# @memory.cache
def read_jpgs(root_path: str, ids: list):
    """read jpg images in the root_path and save them in one numpy array"""
    file_names = (pd.Series(ids).astype(str) + ".jpg").values.tolist()
    imgs = []

    for file in tqdm(file_names):
        _path = str(os.path.join(str(root_path), file))

        img = im.open(_path)
        imgs.append(np.expand_dims(np.asarray(img), axis=0))

    return np.array(imgs)

def get_data_and_labels(data_root, ids, feature_root):
    usecols =["eid", "31-0.0", "22407-2.0", "22408-2.0"]

    img_arrays = read_jpgs(data_root, ids)
    features = pd.read_csv(feature_root, usecols=usecols).set_index('eid').loc[ids]
    features.columns = ["sex", "VAT", "ASAT"]
    # features.insert(features.shape[1], 'VAT/ASAT', features['VAT']/features['ASAT'])
    features = features.dropna()
    features["index"] = np.array(range(features.shape[0]))

    # male_features = features[features["sex"] == 1].drop("sex", axis=1)
    # female_features = features[features["sex"] == 0].drop("sex", axis=1)

    male_data = torch.tensor(img_arrays[features[features["sex"] == 1].drop("sex", axis=1)["index"]]).float()
    female_data = torch.tensor(img_arrays[features[features["sex"] == 0].drop("sex", axis=1)["index"]]).float()
    male_targets = torch.tensor(features[features["sex"] == 1].drop("sex", axis=1).drop("index", axis=1).values).float()
    female_targets = torch.tensor(features[features["sex"] == 0].drop("sex", axis=1).drop("index", axis=1).values).float()

    img_arrays = None
    features = None
    
    return male_data, female_data, male_targets, female_targets

def get_dataloaders(data, targets, batch_size):
    x_train, x_val, y_train, y_val = train_test_split(data, targets, test_size=0.4, random_state=42, shuffle=True)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=42, shuffle=True)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def train(model, trainloader, valloader, config, device, run_n):
    """train model"""
    loss_criterion = RegressionShrinkageLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['base_lr'], weight_decay=config['weight_decay'])
    scheduler = None
    if config['decayed_lr']:
        scheduler = CosineAnnealingLR(optimizer, config['epochs'])

    model.train()

    best_loss = float("+inf")
    train_loss_running = 0.
    training_n = 0

    wandb.init(project = "silhouette-prediction", config = config, name=str(run_n), reinit=True) 

    for epoch in range(config['epochs']):
        for i, train_data in tqdm(enumerate(trainloader)):  
            x_train, y_train = train_data[0], train_data[1]
            x_train = x_train.to(device)
            optimizer.zero_grad()
            prediction = model(x_train)
            y_train = y_train.to(device)
            loss = loss_criterion(prediction, y_train)  
            loss.to(device)
            loss.backward()

            if config['clip_norm'] is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config['clip_norm'])

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            # loss logging
            train_loss_running += loss.item() * y_train.shape[0]
            iteration = epoch * len(trainloader) + i
            training_n += y_train.shape[0]

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
                    x_val, y_val = val_data[0].to(device), val_data[1].to(device)
                    
                    with torch.no_grad():
                        # Get prediction scores
                        prediction = model(x_val)
                                  
                    # y_val = y_val.detach().cpu()
                    #keep track of loss_total_val                                  
                    _predictions = torch.cat((_predictions.float(), prediction.float()))
                    _labels = torch.cat((_labels.float(), y_val.float()))

                accuracy = r2_score(_labels.detach().cpu(), _predictions.detach().cpu(), multioutput='raw_values')
                loss_val = loss_criterion(_predictions, _labels)
                # loss_val = 1.

                print(f'[{epoch:03d}/{i:05d}] val_loss: {loss_val:.3f}')
                wandb.log({"validation loss": loss_val, "epoch": epoch})
                for i, acc in enumerate(accuracy):
                    wandb.log({"val_acc_" + str(i): acc}) 

                if loss_val < best_loss:
                    patience_counter = 0
                    # best_accuracy = accuracy
                    best_loss = loss_val

                    print("new best loss:", best_loss)
                    wandb.log({"best_loss": best_loss}) 

                    for i, best_acc in enumerate(accuracy):
                        wandb.log({"best_val_acc_" + str(i): best_acc}) 

                    torch.save(model, f'./best-models/{config["experiment_name"]}/run{run_n}_model_best.pt')
                    torch.save(model.state_dict(), f'./best-models/{config["experiment_name"]}/run{run_n}_model_best.ckpt')

                # set model back to train
                model.train()


def evaluate(model, loader, device):
    """evaluate model"""
    model.eval()
 
    predictions = torch.tensor([]).to(device)
    labels = torch.tensor([]).to(device)
    
    with torch.no_grad():
        for val_data in loader:
            x_val, y_val = val_data[0].to(device), val_data[1].to(device)
            
            # Get prediction scores
            prediction = model(x_val)

            # y_val = y_val.detach().cpu()
                
            #keep track of loss_total_val                                  
            predictions = torch.cat((predictions.float(), prediction.float()))
            labels = torch.cat((labels.float(), y_val.float()))

        r2 = r2_score(labels.detach().cpu(), predictions.detach().cpu(), multioutput='raw_values')
    return r2





