from pathlib import Path
import glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import open3d as o3d
import nibabel as nib
from skimage.measure import marching_cubes

DATA_ROOT = Path("/vol/chameleon/projects/mesh_gnn/body_segmentations")
TARGET_ROOT = "/vol/chameleon/projects/mesh_gnn/dec_body_25"
VERTICES = 25000
EXTENSION = ".off"

def process(DATA_ROOT, TARGET_ROOT, VERTICES, EXTENSION):

    for file in tqdm(os.listdir(str(DATA_ROOT))):
        _path = str(os.path.join(str(DATA_ROOT), file).replace('\\', '/')) +'/body_mask.nii.gz'
        
        _id = _path[_path[:_path.rfind("/")].rfind("/")+1:_path.rfind("/",0,)]

        body_segment = nib.load(_path)
        body_segment_data = body_segment.get_fdata()
        verts, faces, _, __ = marching_cubes(body_segment_data, level=0, step_size=1)
        verts = verts/np.array(body_segment_data.shape) 

        mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(np.asarray(verts)),
                                    triangles=o3d.utility.Vector3iVector(np.asarray(faces)))

        decimated_mesh = o3d.geometry.TriangleMesh.simplify_quadric_decimation(mesh, VERTICES-1)

        _target_path=  TARGET_ROOT + "/" + _id+ EXTENSION

        o3d.io.write_triangle_mesh( _target_path, decimated_mesh)

def main():
    process(DATA_ROOT, TARGET_ROOT, VERTICES, EXTENSION)

if __name__ == "__main__":
    main()
