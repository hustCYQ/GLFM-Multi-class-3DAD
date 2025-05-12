from data.mvtec3d import get_mvtec_loader,mvtec3d_classes
from data.real3d import get_real_loader,real3d_classes
import torch
from tqdm import tqdm
from feature_extractors.GLFM import GLFM


import numpy as np
import os


class PatchCore():
    
    def __init__(self, args=None):
        self.args = args
        self.method  = GLFM(args=self.args)
        self.dataset_name = self.args.dataset


    def get_dataloader(self,dataset_name,split,class_name):
        if dataset_name == 'mvtec':
            return get_mvtec_loader(split, class_name=class_name)
        if dataset_name == 'real':
            return get_real_loader(split, class_name=class_name)


    def fit(self, class_name):
        train_loader = self.get_dataloader(self.dataset_name,'train',class_name)
        idx = 5
        for pc, _, _, path in tqdm(train_loader, desc=f'Extracting train features for class {class_name}'):
            # pc expected [1,n,3]
            self.method.collect_features(pc)
            self.method.name_list.append(path)
            # idx = idx - 1
            # if idx==0:
            #     break


        print(f'\n\nRunning cluster  for on {class_name} class ...')
        self.method.k_class = self.args.k_class
        self.method.divide_bank()
        self.method.add_sample_to_mem_bank()
        self.method.run_coreset()

    def evaluate(self, class_name):
        image_rocaucs = dict()
        pixel_rocaucs = dict()
        au_pros = dict()
        test_loader = self.get_dataloader(self.dataset_name,'test',class_name)



        with torch.no_grad():
            self.method.init_para()
            self.method.name_list = []
            self.method.test_patch_lib = []
            
            for pc, mask, label, path in tqdm(test_loader, desc=f'Extracting test features for class {class_name}'):
                self.method.name_list.append(path)
                self.method.predict(pc, mask, label, path)

        method_name = "GLFM"

        self.method.calculate_metrics()
        image_rocaucs[method_name] = round(self.method.image_rocauc, 3)
        pixel_rocaucs[method_name] = round(self.method.pixel_rocauc, 3)
        au_pros[method_name] = round(self.method.au_pro, 3)
        print(
            f'Class: {class_name}, {method_name} Image ROCAUC: {self.method.image_rocauc:.3f}, {method_name} Pixel ROCAUC: {self.method.pixel_rocauc:.3f}, {method_name} AU-PRO: {self.method.au_pro:.3f}')
        return image_rocaucs, pixel_rocaucs, au_pros
