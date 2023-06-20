from helper_methods import load_and_split_dataset, evaluate
from torch_geometric.loader import DataLoader
from datasets.in_memory import IMDataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split, KFold

if __name__ == "__main__":
    torch.cuda.set_device(0)
    REGISTERED_ROOT = "/vol/space/projects/ukbb/projects/silhouette/registered_1" # the path of the dir saving the .ply registered data
    INMEMORY_ROOT = '/vol/space/projects/ukbb/projects/silhouette/imdataset/registered1_imdataset' # the root dir path to save all the artifacts ralated of the InMemoryDataset
    FEATURES_PATH = "/vol/space/projects/ukbb/projects/silhouette/ukb668815_imaging.csv"   
    IDS_PATH = "/vol/space/projects/ukbb/projects/silhouette/eids_filtered.npy"
    TARGET = "all"
    model_path = './cv_all_sage_1k/run1_model_best.pt' 
    task = "regression" # regression or classification

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_female = IMDataset(REGISTERED_ROOT, INMEMORY_ROOT, FEATURES_PATH, IDS_PATH, TARGET, 0)
    dataset_male = IMDataset(REGISTERED_ROOT, INMEMORY_ROOT, FEATURES_PATH, IDS_PATH, TARGET, 1)

    female_list = list(dataset_female)
    male_list = list(dataset_male)

    # dev_female, test_female = female_list[0:10906],female_list[10906:-1]
    # dev_male, test_male = female_list[0:10175],female_list[10175:-1]

    dev_female, test_female = train_test_split(female_list, test_size=1/6, random_state=42)
    dev_male, test_male = train_test_split(male_list, test_size=1/6, random_state=42)

    testloader = DataLoader(test_male + test_female, batch_size = 32)
    test_loader_female = DataLoader(test_female, batch_size = 32)
    test_loader_male = DataLoader(test_male, batch_size = 32)

    model = torch.load(f'./runs/cv_all_sage_1k/run3_model_best.pt')
    model.eval()

    config = {
        "task": "regression",
        "num_classes": 2
    }

    _, acc_test_female = evaluate(model, test_loader_female, device, config)
    _, acc_test_male = evaluate(model, test_loader_male, device, config)
    _, acc_test = evaluate(model, testloader, device, config)

    print('R2_test_female: ' + str(np.array(acc_test_female).tolist()))
    print('R2_test_male: ' + str(np.array(acc_test_male).tolist()))
    print('R2_test: ' + str(np.array(acc_test).tolist()))
          

    

