import torch
import torch.nn as nn

class MeshConv(nn.Module):
    """Convolution on a target face and its neighboring faces.
    
    x: a face's feature vectors
    adj: an adjacent face.
    """
    def __init__(self, in_channels, out_channels):
        super(MeshConv, self).__init__()

        self.conv =  nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def __call__(self, x, adj):
        return self.forward(x, adj)

    def forward(self, x, adj):
        x = self.rebuild(x, adj)
        x = self.conv(x)
        return x

    def rebuild(self, data, adj):
        s = data.shape
        re = data.permute(0,2,1)
        batchs = [i[j] for i, j in zip(re,adj)]
        re = torch.stack(batchs)
        
        target = re[:,:,0]
        neighbors =re[:,:,1:]
        alpha = target
        
        midterm = target.unsqueeze(2).repeat(1,1,3,1)
        sums = torch.abs(midterm - neighbors)
        beta = torch.sum(sums, dim=2)
        ag_term = torch.cat([alpha.unsqueeze(2),
                             beta.unsqueeze(2),
                             ],dim=2).reshape(s[0],-1,s[1])
        in_data = ag_term.permute(0,2,1)
        return in_data

