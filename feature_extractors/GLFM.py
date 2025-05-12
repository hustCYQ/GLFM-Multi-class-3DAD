from utils.mvtec3d_util import *
import open3d as o3d
import numpy as np
import torch
from feature_extractors.features import Features
from feature_extractors.models import *
from feature_extractors.pointnet2_utils import *


def unorganized_data_to_organized(organized_pc, none_zero_data_list):
    '''

    Args:
        organized_pc:
        none_zero_data_list:

    Returns:

    '''
    if not isinstance(none_zero_data_list, list):
        none_zero_data_list = [none_zero_data_list]

    for idx in range(len(none_zero_data_list)):
        none_zero_data_list[idx] = none_zero_data_list[idx].squeeze().detach().cpu().numpy()

    organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy() # H W (x,y,z)
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



class GLFM(Features):
    def __init__(self,args=None):
        self.args = args
        super().__init__(args = args)
        self.point_transformer=PointTransformer(group_size=128, num_group=1024,fetch_idx=args.fetch_idx).to('cuda')
        # self.point_transformer.load_model_from_ckpt("feature_extractors/pointmae_pretrain.pth")
        self.point_transformer.load_state_dict(torch.load(args.model_pth))
        self.point_transformer.eval()


    def get_point_features(self,unorganized_pc):

        unorganized_pc = unorganized_pc.squeeze().numpy()
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices,:]).cuda().unsqueeze(dim=0).permute(0, 2, 1)

        unorganized_pc_no_zeros = unorganized_pc_no_zeros.to(torch.float32)

        with torch.no_grad():
            xyz_features, center, ori_idx, center_idx = self.point_transformer(unorganized_pc_no_zeros.contiguous())

        point_feature = interpolating_points(unorganized_pc_no_zeros, center.permute(0,2,1), xyz_features[0]).permute(0,2,1)

        # Part of Group Class
        num_group = 1024
        group_size = 128
        unorganized_pc_no_zeros = unorganized_pc_no_zeros.permute(0,2,1)
        batch_size, num_points, _ = unorganized_pc_no_zeros.contiguous().shape


        # fps the centers out
        center2, center_idx = fps(unorganized_pc_no_zeros.contiguous(), num_group)  # B G 3

        # knn to get the neighborhood
        knn = KNN(k=group_size, transpose_mode=True)
        _, idx = knn(unorganized_pc_no_zeros, center2)  # B G M
        assert idx.size(1) == num_group
        assert idx.size(2) == group_size
        ori_idx = idx
        idx_base = torch.arange(0, batch_size, device=unorganized_pc_no_zeros.device).view(-1, 1, 1) * num_points
        
        idx = idx + idx_base
        idx = idx.view(-1)

        neighborhood = point_feature.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.reshape(batch_size, num_group, group_size, -1).contiguous()

        new_point_feature = torch.mean(neighborhood,-2)

        return new_point_feature,unorganized_pc_no_zeros,unorganized_pc,center2

    def collect_features(self,pcd):
        feature_maps,_,_,_ = self.get_point_features(pcd)
        feature_maps = feature_maps[0]
        self.tmp_patch_lib.append(feature_maps)
        feature_maps = torch.mean(feature_maps,0).unsqueeze(0)
        self.pre_patch_lib.append(feature_maps)



    def divide_bank(self):
        self.pre_patch_lib = torch.cat(self.pre_patch_lib, 0).cpu()
        num_bank = self.k_class
        if num_bank < self.pre_patch_lib.shape[0]:
            self.coreset_idx = self.get_coreset_idx_randomp(self.pre_patch_lib,
                                                            n=int(num_bank),
                                                            eps=self.coreset_eps, )
            self.pre_patch_lib = self.pre_patch_lib[self.coreset_idx].cuda()




    def add_sample_to_mem_bank(self):

        print(len(self.tmp_patch_lib),self.tmp_patch_lib[0].shape)
        for feature_map in self.tmp_patch_lib:

            idx = self.pre_patch_lib.shape[0]
            if self.patch_lib == []:
                self.patch_lib = [[] for i in range(idx)]
            
            dist = torch.cdist(torch.mean(feature_map,0).unsqueeze(0), self.pre_patch_lib)
            min_val, min_idx = torch.min(dist, dim=1)



            self.patch_lib[min_idx].append(feature_map)




    def predict(self, pcd, mask, label,path):
        point_feature,unorganized_pc_no_zeros,organized_pc,center2 = self.get_point_features(pcd)

        patch_list = []
        patch = point_feature[0]

        self.test_patch_lib.append(patch)

        dist = torch.cdist(torch.mean(patch,0).unsqueeze(0), self.pre_patch_lib)
        min_val, min_idx = torch.min(dist, dim=1)

        patch_list.append(patch)

        self.Compute_Anomaly_Score(patch_list, patch.shape[0], mask, label, organized_pc, path,unorganized_pc_no_zeros,center2,bank_idx=min_idx)
  