from helper_methods import load_and_split_dataset, evaluate
from torch_geometric.loader import DataLoader
import torch

if __name__ == "__main__":
    # ------------------configure the metadata for testing-------------
    torch.cuda.set_device(3)
    REGISTERED_ROOT = "/data1/practical-wise2223/registered_5" # the path of the dir saving the .ply registered data
    INMEMORY_ROOT = '/data1/practical-wise2223/registered5_gender_seperation_root' # the root dir path to save all the artifacts ralated of the InMemoryDataset
    FEATURES_PATH = "/vol/chameleon/projects/mesh_gnn/basic_features.csv" # the path of the feature file
    TARGET = "age"
    experiment_name = 'age_prediction_GAT_5k' # the dir path your model is saved
    task = "regression" #regression or classification
    # ------------------------------------------------------------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data_all, val_data_all, test_data_male, test_data_female = load_and_split_dataset(REGISTERED_ROOT, INMEMORY_ROOT, FEATURES_PATH, TARGET)
    train_loader = DataLoader(train_data_all, batch_size = 32, shuffle = True)
    val_loader = DataLoader(val_data_all, batch_size = 32, shuffle = True)
    test_loader_female = DataLoader(test_data_female, batch_size = 32)
    test_loader_male = DataLoader(test_data_male, batch_size = 32)

    model = torch.load(f'./runs/{experiment_name}/model_best.pt')
    model.eval()

    loss_test_female, acc_test_female = evaluate(model, test_loader_female, device, task)
    loss_test_male, acc_test_male = evaluate(model, test_loader_male, device, task)
    ratio_male = len(test_data_male) / (len(test_data_female) + len(test_data_male))
    ratio_female = len(test_data_female) / (len(test_data_female) + len(test_data_male))
    loss_test = loss_test_female * ratio_female + loss_test_male * ratio_male
    acc_test = acc_test_female * ratio_female + acc_test_male * ratio_male

    loss_name = 'MSE' if task == "regression" else 'Crossentropy'
    acc_name = 'R2' if task == "regression" else 'Accuracy'

    print(loss_name + '_test_female: ' + str(loss_test_female.item()))
    print(acc_name + '_test_female: ' + str(acc_test_female.item()))
    print(loss_name + '_test_male: ' + str(loss_test_male.item()))
    print(acc_name + '_test_male: ' + str(acc_test_male.item()))
    print(loss_name + '_test: ' + str(loss_test.item()))
    print(acc_name + '_test: ' + str(acc_test.item()))
          

    

