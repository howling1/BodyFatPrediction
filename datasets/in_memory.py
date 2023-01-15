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
    def __init__(self, raw_data_root, InMemoryDataset_root, basic_features_path, target_name, transform=None, pre_transform=None):
        self.raw_data_root = raw_data_root
        self.basic_features_df = pd.read_csv(basic_features_path)
        self.target_name = target_name
        super(IMDataset, self).__init__(root=InMemoryDataset_root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
 
    @property
    def raw_file_names(self):
        return [os.path.join(str(Path(self.raw_data_root)), file).replace('\\', '/') for file in os.listdir(str(self.raw_data_root))]

    @property
    def processed_file_names(self):
        return self.root + '/' + self.target_name + "_dataset.pt"
 
    def download(self):
        pass
    
    def process(self):
        data_list = []

        for file_path in tqdm(self.raw_file_names):
            _id = os.path.splitext(os.path.basename(file_path))[0]

            if self.target_name == "sex":
                _sex = [[0, 1]] if int(self.basic_features_df["31-0.0"][self.basic_features_df.index[self.basic_features_df['eid'] == int(_id)]]) == 0 else [[1, 0]]# [0,1] = female, [1,0] = male
                _y = torch.tensor(_sex).double()
            elif self.target_name == "BMI":
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
            data_list.append(Data(x=_vertices, edge_index=_edges, y=_y))
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



