"""
PatchCore logic based on https://github.com/rvorias/ind_knn_ad
"""

from sklearn import random_projection
from utils.utils import KNNGaussianBlur
from utils.utils import set_seeds
import numpy as np
from sklearn.metrics import roc_auc_score
import timm
import torch
from tqdm import tqdm
from utils.au_pro_util import calculate_au_pro
from feature_extractors.pointnet2_utils import *

import cv2
import os
# from utils.visz_utils import *

def organized_pc_to_unorganized_pc(organized_pc):
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])




def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


class Features(torch.nn.Module):

    def __init__(self, image_size=224, f_coreset=0.1, coreset_eps=0.9,args = None):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.image_size = image_size
        self.f_coreset = f_coreset
        self.coreset_eps = coreset_eps
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.blur = KNNGaussianBlur(4)
        self.n_reweight = 3
        self.patch_lib = []
        self.pre_patch_lib = []
        self.tmp_patch_lib = []
        self.name_list = []
        self.test_patch_lib = []

        self.resize = torch.nn.AdaptiveAvgPool2d((28, 28))
        self.resize2 = torch.nn.AdaptiveAvgPool2d((14, 14))

        self.image_preds = list()
        self.image_labels = list()
        self.pixel_preds = list()
        self.pixel_labels = list()
        self.gts = []
        self.predictions = []
        self.image_rocauc = 0
        self.pixel_rocauc = 0
        self.au_pro = 0

    def __call__(self, x):
        # Extract the desired feature maps using the backbone model.
        with torch.no_grad():
            feature_maps = self.deep_feature_extractor(x)

        feature_maps = [fmap.to("cpu") for fmap in feature_maps]
        return feature_maps


    def unorganized_data_to_organized(self,organized_pc, none_zero_data_list):
        '''

        Args:
            organized_pc:
            none_zero_data_list:

        Returns:

        '''
        # print(none_zero_data_list[0].shape)
        if not isinstance(none_zero_data_list, list):
            none_zero_data_list = [none_zero_data_list]

        for idx in range(len(none_zero_data_list)):
            none_zero_data_list[idx] = none_zero_data_list[idx].squeeze().detach().cpu().numpy()

        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy() # H W (x,y,z)
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]


        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]



        full_data_list = []

        for none_zero_data in none_zero_data_list:
            if none_zero_data.ndim == 1:
                none_zero_data = np.expand_dims(none_zero_data,1)
            full_data = np.zeros((unorganized_pc.shape[0], none_zero_data.shape[1]), dtype=none_zero_data.dtype)
            full_data[nonzero_indices, :] = none_zero_data
            full_data_reshaped = full_data.reshape((organized_pc_np.shape[0], organized_pc_np.shape[1], none_zero_data.shape[1]))
            full_data_tensor = torch.tensor(full_data_reshaped).permute(2, 0, 1).unsqueeze(dim=0)
            full_data_list.append(full_data_tensor)

        return full_data_list

    def add_sample_to_mem_bank(self, sample):
        raise NotImplementedError

    def predict(self, sample, mask, label):
        raise NotImplementedError

    def init_para(self):
        self.image_preds = list()
        self.image_labels = list()
        self.pixel_preds = list()
        self.pixel_labels = list()
        self.gts = []
        self.predictions = []
        self.image_rocauc = 0
        self.pixel_rocauc = 0
        self.au_pro = 0


    def Compute_Anomaly_Score(self, patch_list, feature_map_dims, mask, label,organized_pc, path, unorganized_pc_no_zeros,center2,bank_idx = 0):
            
        n = len(patch_list)
        patch = patch_list[0]


        dist = torch.cdist(patch, self.patch_lib[bank_idx])
        min_val, min_idx = torch.min(dist, dim=1)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)


        s_map = min_val.view(1, 1, feature_map_dims)
        s_map = interpolating_points(unorganized_pc_no_zeros.permute(0,2,1), center2.permute(0,2,1), s_map).permute(0,2,1)
        
        if self.args.dataset == 'mvtec':
            organized_pc = torch.tensor(organized_pc).squeeze().permute(1,0)
            organized_pc = organized_pc.reshape(1,organized_pc.shape[0],224,224)
            s_map = torch.Tensor(self.unorganized_data_to_organized(organized_pc, [s_map])[0])
            s_map = s_map.squeeze().reshape(1,224,224)
            s_map = self.blur(s_map)
            s_map = s_map.cpu()
            s = torch.max(s_map).cpu()
        if self.args.dataset == 'real':
            s_map = s_map.cpu()
            # s = torch.max(s_map).cpu()
            s = torch.mean(s_map).cpu()
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())









    def calculate_metrics(self):
        self.image_preds = np.stack(self.image_preds)
        self.image_labels = np.stack(self.image_labels)
        self.pixel_preds = np.array(self.pixel_preds)

        self.image_rocauc = roc_auc_score(self.image_labels, self.image_preds)
        self.pixel_rocauc = roc_auc_score(self.pixel_labels, self.pixel_preds)
        if self.args.dataset == 'mvtec' or self.args.dataset == 'eyecandies':
            self.au_pro, _ = calculate_au_pro(self.gts, self.predictions)
        if self.args.dataset == 'real' or self.args.dataset == 'shapenet' or self.args.dataset == 'mulsen':
            self.au_pro = 0





    def run_coreset(self):
        n = len(self.patch_lib)
        for i in range(n):
            self.patch_lib[i] = torch.cat(self.patch_lib[i], 0).cpu()
            if self.f_coreset < 1:
                self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib[i],
                                                                n=int(self.f_coreset * self.patch_lib[i].shape[0]),
                                                                eps=self.coreset_eps, )
                self.patch_lib[i] = self.patch_lib[i][self.coreset_idx].cuda()

    def get_coreset_idx_randomp(self, z_lib, n=1000, eps=0.90, float16=True, force_cpu=False):
        """Returns n coreset idx for given z_lib.
        Performance on AMD3700, 32GB RAM, RTX3080 (10GB):
        CPU: 40-60 it/s, GPU: 500+ it/s (float32), 1500+ it/s (float16)
        Args:
            z_lib:      (n, d) tensor of patches.
            n:          Number of patches to select.
            eps:        Agression of the sparse random projection.
            float16:    Cast all to float16, saves memory and is a bit faster (on GPU).
            force_cpu:  Force cpu, useful in case of GPU OOM.
        Returns:
            coreset indices
        """

        print(f"   Fitting random projections. Start dim = {z_lib.shape}.")
        try:
            transformer = random_projection.SparseRandomProjection(eps=eps)
            z_lib = torch.tensor(transformer.fit_transform(z_lib))
            print(f"   DONE.                 Transformed dim = {z_lib.shape}.")
        except ValueError:
            print("   Error: could not project vectors. Please increase `eps`.")

        select_idx = 0
        last_item = z_lib[select_idx:select_idx + 1]
        coreset_idx = [torch.tensor(select_idx)]
        min_distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)
        # The line below is not faster than linalg.norm, although i'm keeping it in for
        # future reference.
        # min_distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True)

        if float16:
            last_item = last_item.half()
            z_lib = z_lib.half()
            min_distances = min_distances.half()
        if torch.cuda.is_available() and not force_cpu:
            last_item = last_item.to("cuda")
            z_lib = z_lib.to("cuda")
            min_distances = min_distances.to("cuda")

        for _ in tqdm(range(n - 1)):
            distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)  # broadcasting step
            min_distances = torch.minimum(distances, min_distances)  # iterative step
            select_idx = torch.argmax(min_distances)  # selection step

            # bookkeeping
            last_item = z_lib[select_idx:select_idx + 1]
            min_distances[select_idx] = 0
            coreset_idx.append(select_idx.to("cpu"))
        return torch.stack(coreset_idx)

