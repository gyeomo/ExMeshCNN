import torch
import torch.nn as nn

class Geometric(nn.Module):
    """geometric convolution on a target face and its neighboring faces. 
    
    x: it unifies a centroid and a normal vector of a face.
    """
    def __init__(self, middle_channels, out_channels):
        super(Geometric, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=middle_channels, kernel_size=4, stride=4),
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
        """
        t_normal, n_normal: a face's normal vector
        t_center, n_center: a face's centroid
        """
        target = data[:,:,0]
        neighbors =data[:,:,1:]
        t_point = target[:,:,:9]
        t_normal = target[:,:,9:]
        n_point = neighbors[:,:,:,:9]
        n_normal = neighbors[:,:,:,9:]
        t_center = (t_point[:,:,:3] + t_point[:,:,3:6] + t_point[:,:,6:])/3
        n_center = (n_point[:,:,:,:3] + n_point[:,:,:,3:6] + n_point[:,:,:,6:])/3
        
        first = t_center
        second = t_normal

        midterm1 = t_center.unsqueeze(2).repeat(1,1,3,1)
        sums1 = torch.abs(midterm1 - n_center)
        third = torch.sum(sums1, dim=2)

        midterm2 = t_center.unsqueeze(2).repeat(1,1,3,1)
        sums2 = torch.abs(midterm2 - n_center)
        fourth = torch.sum(sums2, dim=2)       
        
        ag_term = torch.cat([first.unsqueeze(2),
                             second.unsqueeze(2),
                             third.unsqueeze(2),
                             fourth.unsqueeze(2),
                             ],dim=2)
        ag_term = ag_term.reshape(data.shape[0], -1, 3)
        in_data = ag_term.permute(0,2,1).float()
        return in_data
