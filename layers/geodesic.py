import torch
import torch.nn as nn

class Geodesic(nn.Module):
    """geodesic convolution based on the vertices and edges aligned with a target face and its neighboring faces. 
    
    x: it unifies an edge path and a normal vector of vertices
    """
    def __init__(self, middle_channels, out_channels):
        super(Geodesic, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=middle_channels, kernel_size=3, stride=3),
            nn.BatchNorm1d(middle_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=middle_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.rebuild(x)
        x = self.conv(x)
        return x

    def rebuild(self, data):
        # edge path between a target face's vertices and their adjacent vertices 
        edges_c = torch.abs(data[:,:,:,:3] - data[:,:,:,3:6])
        # A path between a target face's center point and its vertices
        faedge =  data[:,:,:,6:9]
        # Normal vectors of a target face's vertices
        norms =  data[:,:,:,9:]
        
        first = torch.sum(edges_c,2).unsqueeze(2)
        second = torch.sum(faedge,2).unsqueeze(2)
        third = torch.sum(norms,2).unsqueeze(2)
        re_feature = torch.cat([first, 
                                second,
                                third,
                                ], dim=2).reshape(edges_c.shape[0],-1,3)
        in_data = re_feature.permute(0,2,1).float()
        return in_data
