import torch.nn as nn
import torch
from layers.geodesic import Geodesic
from layers.geometric import Geometric
from layers.meshconv import MeshConv

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

    def forward(self, ed, fa, ad):
        ed = self.conv_e(ed)
        fa = self.conv_f(fa)
        fe = torch.cat([ed,fa],dim=1)
        fe = self.conv1(fe, ad)
        fe = self.conv2(fe, ad)
        fe = self.conv3(fe, ad)
        fe = self.conv4(fe, ad)
        fe = self.fcn(fe)
        fe = self.avg_pool(fe)
        fe = fe.view(fe.size(0), -1)
        return fe

