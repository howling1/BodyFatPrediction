## About the Project
This repo is the result of our experiments through the guided research project Body Fat Prediction using GNN at TUM. Purpose is to explore different GNN architectures on triangular meshes of human bodies to predict specific fat depot, including VAT and ASAT. Data source is the UKBioBank. Pre-process includes decimation, registration and creating in memory dataset to keep target feature with data. Architectures we have tested: GAT, FeaStNet, SAGE, GCN, ResGNN, DenseGNN, JKNet, Graphormer

## File description
`datasets/` includes the class for InMemoryDataset initialization.\
`notebooks/` includes exploratory analysis Jupyter Notebook files. Visualizations of the meshes and some statistics for the data can be seen there.\
`models/` includes different GNN architectures\
`preprocess/` includes the scripts for decimation and registration processes\
`runs/` folder to keep the model checkpoints (empty)\
`helper_methods.py` has functions for utilization\
`train_cv.py` is to experiment with different models and parameters with 5-fold cross validation\

## How to run the code
`Generating mesh from segmentation from scratch:` run the `preprocess/decimate.py` and `preprocess/register.py` in sequence on the segmentation data.
`Train and test model:` set the path of registered meshes`REGISTERED_ROOT`, the path for saving the IMDataset`INMEMORY_ROOT`, the features csv`FEATURES_PATH`, the eid file path`IDS_PATH` and the target you want to predict`TARGET` in the `train_cv.py` and run this script. Then the script will generate an IMDataset on the server, which will be used to load data faster for following training, and start a 5-fold cross validation training. You can change the hparams and models you want in the script for optimization. The thing to note is that you need to make sure the number of nodes of all meshes is the same when you use the Graphormer model.







