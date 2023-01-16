from pathlib import Path
import open3d as o3d
import os
from tqdm import tqdm
import numpy as np
import math
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data

class IMDataset(InMemoryDataset):
 
    def __init__(self, raw_data_root, dataset_root, basic_features_path, target_name, gender, transform=None, pre_transform=None):
        self.raw_data_root = raw_data_root
        self.basic_features_df = pd.read_csv(basic_features_path)
        self.target_name = target_name
        super(IMDataset, self).__init__(root=dataset_root, transform=transform, pre_transform=pre_transform)
        if gender == 0: #female
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif gender == 1: #male
            self.data, self.slices = torch.load(self.processed_paths[1])
 
    @property
    def raw_file_names(self):
        return [os.path.join(str(Path(self.raw_data_root)), file).replace('\\', '/') for file in os.listdir(str(self.raw_data_root))]

    @property
    def processed_file_names(self):
        processed_root = self.root + "/" + self.target_name

        if not os.path.exists(processed_root):
            os.makedirs(processed_root)

        return [processed_root + "/female_dataset.pt", processed_root + "/male_dataset.pt"]
 
    def download(self):
        pass
    
    def process(self):
        male_data_list = []
        female_data_list = []

        for file_path in tqdm(self.raw_file_names):
            _id = os.path.splitext(os.path.basename(file_path))[0]
            _sex = int(self.basic_features_df["31-0.0"][self.basic_features_df.index[self.basic_features_df['eid'] == int(_id)]]) #female: 0, male: 1
                
            if self.target_name == "sex":
                _y = torch.tensor([[0, 1]] if _sex == 0 else [[1, 0]]).double() # [0,1] = female, [1,0] = male
            elif self.target_name == "bmi":
                _bmi = self.basic_features_df["21001-2.0"][self.basic_features_df.index[self.basic_features_df['eid'] == int(_id)]].values
                if math.isnan(_bmi[0]):
                    continue
                else:
                    _y = torch.tensor(_bmi).double()
            elif self.target_name == "height":
                _height = self.basic_features_df["50-2.0"][self.basic_features_df.index[self.basic_features_df['eid'] == int(_id)]].values
                if math.isnan(_height[0]):
                    continue
                else:
                    _y = torch.tensor(_height).double()
            elif self.target_name == "weight":
                _weight = self.basic_features["21002-2.0"][self.basic_features.index[self.basic_features['eid'] == int(_id)]].values
                if math.isnan(_weight[0]):
                    continue
                else:
                    _y = torch.tensor(_weight).double()
            elif self.target_name == "age":
                _age = self.basic_features["21003-2.0"][self.basic_features.index[self.basic_features['eid'] == int(_id)]].values
                if math.isnan(_age[0]):
                    continue
                else:
                    _y = torch.tensor(_age).double()

            _mesh = o3d.io.read_triangle_mesh(file_path)
            _vertices = torch.from_numpy(np.asarray(_mesh.vertices)).double()
            _triangles = np.asarray(_mesh.triangles)

            edge_list = []
            for t in _triangles:
                edge_list.append([t[0], t[1]])
                edge_list.append([t[1], t[0]])
                edge_list.append([t[0], t[2]])
                edge_list.append([t[2], t[0]])
                edge_list.append([t[1], t[2]])
                edge_list.append([t[2], t[1]])

            _edges = torch.from_numpy(np.unique(np.array(edge_list), axis=0).reshape(2,-1)).long()

            if _sex == 0:
                female_data_list.append(Data(x=_vertices, edge_index=_edges, y=_y))
            else:
                male_data_list.append(Data(x=_vertices, edge_index=_edges, y=_y))
        
        data_female, slices_female = self.collate(female_data_list)
        data_male, slices_male = self.collate(male_data_list)
        
        torch.save((data_female, slices_female), self.processed_paths[0])
        torch.save((data_male, slices_male), self.processed_paths[1])