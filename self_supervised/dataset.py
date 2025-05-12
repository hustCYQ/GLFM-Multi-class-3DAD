import os
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
# from utils.mvtec3d_util import *
from torch.utils.data import DataLoader
import numpy as np

import random
from self_supervised.generateAD import *



DATASETS_PATH = '../Semicore/data/mvtec3ds'

def mvtec3d_classes():
    return [
        "bagel",
        "cable_gland",
        "carrot",
        "cookie",
        "dowel",
        "foam",
        "peach",
        "potato",
        "rope",
        "tire",
    ]


class Generate3D_AD(Dataset):

    def __init__(self, split='train', img_size=224, p=0.5):
        super().__init__()
        self.size = img_size
        self.p = p 

        self.tiff_tot_paths= self.load_dataset()  

    def load_dataset(self):
        tiff_tot_paths = []
        for name in mvtec3d_classes():
            self.img_path = os.path.join(DATASETS_PATH, name, 'train')
            tiff_paths = glob.glob(os.path.join(self.img_path, 'good', 'xyz') + "/*.tiff")
            tiff_paths.sort()
            tiff_tot_paths.extend(tiff_paths)

        return tiff_tot_paths

    def __len__(self):
        return len(self.tiff_tot_paths)

    def __getitem__(self, idx):

        tiff_path = self.tiff_tot_paths[idx]
        organized_pc = read_tiff_organized_pc(tiff_path)
        organized_pc = resize_organized_pc(organized_pc).permute(1,2,0)
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc).numpy()
        
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]

        if random.randint(0,100)<=100*self.p:
            pcd, mask = stretching(unorganized_pc_no_zeros)
            label = 1
      
        pcd = torch.Tensor(pcd)
        mask = torch.Tensor(mask)

        return pcd,mask,label
