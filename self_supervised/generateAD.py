# 1. 读取点云
# 2. 计算法矢
# 3. 选择点，计算点的邻域
# 4. 将点及邻域向法矢方向拉伸
# 5. 保存为新的点

import numpy as np
import random
import open3d as o3d
import time

import tifffile as tiff
import torch


def organized_pc_to_unorganized_pc(organized_pc):
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])


def read_tiff_organized_pc(path):
    tiff_img = tiff.imread(path)
    return tiff_img


def resize_organized_pc(organized_pc, target_height=224, target_width=224, tensor_out=True):
    torch_organized_pc = torch.tensor(organized_pc).permute(2, 0, 1).unsqueeze(dim=0)
    torch_resized_organized_pc = torch.nn.functional.interpolate(torch_organized_pc, size=(target_height, target_width),
                                                                 mode='nearest')
    if tensor_out:
        return torch_resized_organized_pc.squeeze(dim=0)
    else:
        return torch_resized_organized_pc.squeeze().permute(1, 2, 0).numpy()


def organized_pc_to_depth_map(organized_pc):
    return organized_pc[:, :, 2]


def stretching(unorganized_pc_no_zeros):

    tot_point_num = unorganized_pc_no_zeros.shape[0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(unorganized_pc_no_zeros[:,0:3]) 

    radius = 1000000 
    max_nn = 15  
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))


    random.seed(time.time())


    selected_point_num = random.randint(int(tot_point_num/100), int(tot_point_num/50))

    selected_idx = random.randint(0, tot_point_num-1)
    direction = random.randint(-100,100)
    if direction>=0:
        direction = 1 
    else:
        direction=-1

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)  
    dis_list = []
    for idx in range(0,tot_point_num):
        point = pcd.points[idx]   
        k = 2   
        [k, idx, dist] = pcd_tree.search_knn_vector_3d(point, k) 
        dis_list.append(dist[1])
    dis_points = np.array(dis_list).mean()
    

    dx = random.randint(80, 120)*1.0/100
    dy = random.randint(80, 120)*1.0/100
    dz = random.randint(80, 120)*1.0/100

    pcd_s =  o3d.geometry.PointCloud()
    pcd_s.points = pcd.points
    for i in range(0,tot_point_num):
        pcd_s.points[i][0] = pcd_s.points[i][0] * dx
        pcd_s.points[i][1] = pcd_s.points[i][1] * dy
        pcd_s.points[i][2] = pcd_s.points[i][2] * dz
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_s)

    point = pcd.points[selected_idx]  

    k = selected_point_num 
    [k, idx, dist] = pcd_tree.search_knn_vector_3d(point, k) 

    mask = np.zeros(tot_point_num,dtype=int)
    for i in range(0,k):
        pcd.points[idx[i]] = pcd.points[idx[i]] + pcd.normals[selected_idx] * (k-i) * dis_points * 100 * direction
        mask[idx[i]] = 1


    return np.asarray(pcd.points),mask
