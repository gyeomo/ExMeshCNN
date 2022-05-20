import numpy as np
import torch
import torch.nn as nn
import os
import time
from layers.geodesic import Geodesic
from layers.geometric import Geometric
from layers.meshconv import MeshConv

import openmesh as om
import pymeshlab as ml
import trimesh as tm
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm as cmx

root_dir = './dataset' # dataset name for training
original_dataset = './cubes_500' # manifold dataset name (results of resize_manifold.py)

class SingleImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, mode):
        self.root_dir = root_dir
        class_names = sorted(os.listdir(self.root_dir))
        self.classnames = class_names
        self.mode = mode
        
        self.filepaths = []
        for class_name in class_names:
            path = os.path.join(root_dir, class_name, mode)
            obj_name = np.sort(os.listdir(path))
            for obj in obj_name:
                obj_path = os.path.join(path, obj)
                self.filepaths.append(obj_path)
        
    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('/')[2]
        class_id = self.classnames.index(class_name)
        
        d_1 = np.load(self.filepaths[idx], allow_pickle=True)
        ed = torch.from_numpy(d_1.item()['edge'])
        fa = torch.from_numpy(d_1.item()['face'])
        ad = d_1.item()['adj']
        return (class_id, ed , fa, ad, path)


class ExMeshCNN(nn.Module):
    """
    ed: edge feature
    fa: face feature
    ad: adjacent face list
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv_e = Geodesic(128,64)
        self.conv_f = Geometric(128,64)
        self.conv1 = MeshConv(128,128)
        self.conv2 = MeshConv(128,256)
        self.conv3 = MeshConv(256,256)
        self.conv4 = MeshConv(256,512)
        self.fcn = nn.Sequential(
            nn.Conv1d(in_channels=512 , out_channels=num_classes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(num_classes)
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self):
        return self.acts

    def forward(self, ed, fa, ad):
        ed = self.conv_e(ed)
        fa = self.conv_f(fa)
        fe = torch.cat([ed,fa],dim=1)
        fe = self.conv1(fe, ad)
        fe = self.conv2(fe, ad)
        fe = self.conv3(fe, ad)
        fe = self.conv4(fe, ad)
        self.acts = fe
        h = fe.register_hook(self.activations_hook)
        fe = self.fcn(fe)
        fe = self.avg_pool(fe)
        fe = fe.view(fe.size(0), -1)
        return fe

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
class_names = sorted(os.listdir(root_dir)) 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 0.001
training_epochs = 500
batch_size = 16

model = ExMeshCNN(len(class_names)).to(device)

model.load_state_dict(torch.load("./model_save.pt"))
model.eval();

val_dataset = SingleImgDataset(root_dir, 'test')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

for k, data in enumerate(val_loader):
    print("({}/{})".format(len(val_loader), k+1))
    Y = data[0].to(device)
    X1 = data[1].to(device)
    X2 = data[2].to(device)
    X3 = data[3].to(device)
    pred = model(X1, X2, X3)
    hc = pred.argmax()

    pred[:,hc].backward()
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2])
    activations = model.get_activations()
    for i in range(activations.shape[1]):
        activations[:,i,:] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap -= heatmap.mean()
    heatmap = torch.relu(heatmap)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.detach().cpu()
    path = data[4][0][:-4]
    path = path.replace(root_dir, original_dataset)
    ccmp = plt.get_cmap('Reds')
    cNorm  = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=ccmp)

    mesh = om.read_trimesh(path)
    for i,face in enumerate(mesh.faces()):
        mesh.set_color(face, scalarMap.to_rgba(heatmap[i]))
        
    new_path = path.replace(original_dataset, "Explain")
    path_ = new_path.split("/")[:-1]
    path_dir = ""
    for p in path_:
        path_dir = os.path.join(path_dir,p)
    path_dir
    os.makedirs(path_dir, exist_ok=True)
    
    om.write_mesh(new_path, mesh, face_color=True)


