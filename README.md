## About the Project
This repo is the result of our experiments through the praktikum ADLM at TUM. Purpose is to explore different GNN architectures on graph data, triangular meshes to be exact. Data source is the UKBioBank.
Pre-process includes decimation, registration and creating in memory dataset to keep target feature with data.
Tasks that are avaliable : Sex, BMI, Height, Weight and Age prediction.
Architectures we have tested: GAT, FeaStNet, SAGE, GCN

`datasets/` includes the class for InMemoryDataset initialization.\
`notebooks/` includes exploratory analysis Jupyter Notebook files. Visualizations of the meshes and some statistics for the data can be seen there.\
`models/` includes different GNN architectures\
`preprocess/` includes the scripts for decimation and registration processes\
`runs/` folder to keep the model checkpoints (empty)\

`helper_methods.py` has functions for utilization\
`testing.py` is to test a trained model\
`training.py` is to experiment with different models and parameters\
`training_sweep.py` is for hyperparameter optimization with Wandb Sweep\