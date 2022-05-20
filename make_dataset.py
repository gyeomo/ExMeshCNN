import trimesh as tm
import torch
import numpy as np
import os
from utils.functions import *

# Data pre-processing towards making the entire training easier.
dataset = "./dataset"
database = './cubes' # The number of fases for each mesh must be the same with each other. SWs such as MeshLab can be used.

# ModelNet40, Manifold40: 1024
# Cube, Shrec: 500
target_face_num = 500 

database = database+"_"+str(target_face_num)

class_names = np.sort(os.listdir(database))
modes = ["train", "test"]

for name in class_names:
    for mode in modes:
        path1 = os.path.join(database, name, mode)
        obj_names = os.listdir(path1)
        for obj in obj_names:
            path2 = os.path.join(path1, obj)
            mesh = tm.load_mesh(path2, process=False)
            
            verts = mesh.vertices
            faces = mesh.faces
            
            # normalization
            centroid = mesh.centroid
            verts = verts - centroid
            max_len = np.max(verts[:, 0]**2 + verts[:, 1]**2 + verts[:, 2]**2)
            verts /= np.sqrt(max_len)
            mesh = tm.Trimesh(verts, faces, process=False)

            verts = mesh.vertices
            faces = mesh.faces
            norms = mesh.vertex_normals
            
            faces_t = torch.from_numpy(faces)
            verts_t = torch.from_numpy(verts)

            adjs = torch.from_numpy(mesh.face_adjacency)
            adj_list = get_adj_nm(adjs).long()  
            
            # extract edge feature
            norm_t = torch.from_numpy(norms[faces_t[adj_list[:,0]]])
            in_face = faces_t[adj_list].clone()
            size = len(in_face)
            edges_t = torch.stack([get_edges(in_face[i], verts_t) for i in range(size)])
            edges_t = edges_t.reshape(-1,3,6)
            face_centroid = torch.mean(verts_t[faces_t[adj_list[:,0]]],dim=1)
            facen=face_centroid.unsqueeze(1).repeat(1,3,1)
            facened = facen - verts_t[faces_t[adj_list[:,0]]]
            edge_feature = torch.cat([edges_t, facened, norm_t],dim=2)
            edge_feature = np.float16(edge_feature.detach().numpy())
            
            # extract face feature
            adj_list = adj_list.detach().numpy()
            normals = mesh.face_normals[adj_list]
            faces_t = faces_t.detach().numpy()
            verts_t = verts_t.detach().numpy()
            points = verts_t[faces_t[adj_list]].reshape(-1,4,9)
            face_feature = np.concatenate((points, normals), axis=2)
            
            if (len(edge_feature) == target_face_num) and (len(adj_list) == target_face_num):
                d1={'edge':edge_feature, 'adj':adj_list, 'face':face_feature}
                save_path = os.path.join(dataset, name, mode)
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, obj)
                np.save(save_path, d1)
