from pathlib import Path
import glob
import numpy as np
from tqdm import tqdm
import open3d as o3d


DATA_ROOT = Path("/data1/practical-wise2223/decimated_25")
TARGET_ROOT = "/data1/practical-wise2223/registered"
TARGET_SAMPLE = "/data1/practical-wise2223/4266049_5/4266049_25.ply"
THRESHOLD = 0.02
EXTENSION = ".ply"

def registering_ICP(DATA_ROOT, TARGET_ROOT, TARGET_SAMPLE, THRESHOLD, EXTENSION):
    """
    Function to decimate the mesh data in the given folder
    :param DATA_ROOT: path for the input mesh data 
    :param TARGET_ROOT: path where the registered data will be saved
    :param TARGET_SAMPLE: path to the target mesh that'll be used in the ICP
    :param THRESHOLD: threshold for ICP 
    :param EXTENSION: extension of the files that will be saved
    """
    for file in tqdm(glob.glob(str(DATA_ROOT / "*.ply"))):
        _path = file.replace('\\', '/')

        _id = _path[_path[:_path.rfind(".")].rfind("/")+1:_path.rfind(".",0,)]

        # reading target as triangle mesh and point cloud
        target_trm = o3d.io.read_triangle_mesh(TARGET_SAMPLE)
        target_trm.compute_vertex_normals()
        target_pcd = o3d.geometry.PointCloud(points = target_trm.vertices)
        target_pcd.estimate_normals()

        # reading source as triangle mesh
        source_trm = o3d.io.read_triangle_mesh(_path)
        # creating a point cloud for ICP from triangle mesh
        source_pcd = o3d.geometry.PointCloud(points = source_trm.vertices)
        source_pcd.estimate_normals()
        
        # identity transformation
        trans_init = np.identity(4)
        
        # find transformation for registering
        reg_p2l = o3d.pipelines.registration.registration_icp(source_pcd, target_pcd, THRESHOLD, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPlane())

        # apply transformation to source point cloud
        source_pcd.transform(reg_p2l.transformation)

        # create new triangle mesh object from transformed source point cloud and source triangle mesh face data
        new_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(np.asarray(source_pcd.points)),
                                             triangles=o3d.utility.Vector3iVector(np.asarray(source_trm.triangles)))
        new_mesh.compute_vertex_normals()

        # write the registered mesh 
        _target_path =  TARGET_ROOT + "/" + _id + EXTENSION
        o3d.io.write_triangle_mesh( _target_path, new_mesh)


def main():
    
    registering_ICP(DATA_ROOT, TARGET_ROOT, TARGET_SAMPLE, THRESHOLD, EXTENSION)

if __name__ == "__main__":
    main()
