import numpy as np
import torch
import os

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
        return (class_id, ed , fa, ad)

