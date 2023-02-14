from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import open3d as o3d
import nibabel as nib
from skimage.measure import marching_cubes

DATA_ROOT = Path("/vol/chameleon/projects/mesh_gnn/body_segmentations")
TARGET_ROOT = "/data1/practical-wise2223/decimated_5"
FACES = 5000
EXTENSION = ".ply"
LIMIT = 10000
COUNT = 0

def process(DATA_ROOT, TARGET_ROOT, FACES, EXTENSION, LIMIT, COUNT):
    """
    Function to decimate the mesh data in the given folder
    :param DATA_ROOT: path for the input mesh data 
    :param TARGET_ROOT: path where the decimated data will be saved
    :param FACES: target number of faces to be decimated
    :param EXTENSION: extension of the files that will be saved
    :param LIMIT: how many files to process
    :param COUNT: 0
    """
    for file in tqdm(os.listdir(str(DATA_ROOT))):
        _path = str(os.path.join(str(DATA_ROOT), file).replace('\\', '/')) +'/body_mask.nii.gz'
        
        _id = _path[_path[:_path.rfind("/")].rfind("/")+1:_path.rfind("/",0,)]
        
        if (LIMIT > COUNT) :
            if(os.path.exists(_path)):
            
                body_segment = nib.load(_path)
                body_segment_data = body_segment.get_fdata()

                verts, faces, _, __ = marching_cubes(body_segment_data, level=0, step_size=1)
                verts = verts/np.array(body_segment_data.shape) 

                mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(np.asarray(verts)),
                                            triangles=o3d.utility.Vector3iVector(np.asarray(faces)))

                decimated_mesh = o3d.geometry.TriangleMesh.simplify_quadric_decimation(mesh, FACES)

                _target_path =  TARGET_ROOT + "/" + _id + EXTENSION
                o3d.io.write_triangle_mesh( _target_path, decimated_mesh)
                COUNT += 1
            else:
                continue
        else:
            break

def main():
    process(DATA_ROOT, TARGET_ROOT, FACES, EXTENSION, LIMIT, COUNT)

if __name__ == "__main__":
    main()
