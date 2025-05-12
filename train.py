from self_supervised.dataset import Generate3D_AD


import open3d as o3d
import numpy as np
import torch
from feature_extractors.models import *
from feature_extractors.pointnet2_utils import *

# from data.mvtec3d import get_data_loader
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from loss import *

def train():

    point_transformer=PointTransformer(group_size=128, num_group=1024,fetch_idx=[11]).to('cuda')
    point_transformer.load_model_from_ckpt("feature_extractors/pointmae_pretrain.pth")
    point_transformer.train()


    # loss
    criterion = nn.CrossEntropyLoss()    
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    
    optimizer = torch.optim.Adam(point_transformer.parameters(),lr=0.0001)
    best_acc = 0
    for epoch in range(0,100):
        if epoch == 40:
            for params in optimizer_Adam.param_groups:                    
                params['lr'] *= 0.00001     
        if epoch == 80:
            for params in optimizer_Adam.param_groups:                    
                params['lr'] *= 0.000001     
        iter_step = 100
        running_loss = 0
        running_corrects = 0
        point_pred = []
        point_gt = []
        loss = 0

        train_dataset  = Generate3D_AD(p=1.0)
        train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False,
                                pin_memory=True)

        for pcd, mask, label in tqdm(train_loader, desc=f'class task training'):
            iter_step = iter_step - 1
            if iter_step<0:break

            pcd = pcd.cuda().permute(0, 2, 1)
            mask = mask.cuda()
            mask = mask.to(torch.long)
            mask = mask.squeeze(0).unsqueeze(-1)
            label = label.cuda()

            xyz_features, center, ori_idx, center_idx = point_transformer(pcd.contiguous())
            point_feature = interpolating_points(pcd, center.permute(0,2,1), xyz_features[0]).permute(0,2,1)
            mean_xyz_features = torch.mean(xyz_features[0],-1)
            score_map = point_transformer.do_classify(point_feature).permute(0,2,1)

            
        
            # optimizer
            optimizer.zero_grad()
            mask = mask.permute(1,0)

            loss += loss_focal(score_map,mask.unsqueeze(0).squeeze(-1))  #[1,2,n]  [1,1,n,1]
            loss += loss_dice(score_map[:,1,:],mask.unsqueeze(0).squeeze(-1))

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loss = 0
            
            _, preds = torch.max(score_map, 1)
            running_corrects += (torch.sum(preds == mask) / preds.shape[-1])

            if point_pred == []:
                point_pred = score_map.cpu()[:,1,:].detach()
                point_gt = mask.cpu().detach()
            else:
                point_pred = torch.cat((point_pred,score_map.cpu().detach()[:,1,:]),-1)
                point_gt = torch.cat((point_gt,mask.cpu().detach()),1)    
        
            
        epoch_loss = running_loss / 100
        epoch_acc = running_corrects.double() / 100
        point_pred = point_pred.numpy().ravel()
        point_gt = point_gt.numpy().ravel()
        auroc = roc_auc_score(point_gt, point_pred)
        print('{}  {} Loss: {:.4f} Acc: {:.4f} auroc: {:.4f} '.format(epoch, 'train', epoch_loss, epoch_acc,auroc))

        from sklearn import metrics
        import pylab as plt
        fpr, tpr, threshold = metrics.roc_curve(point_gt, point_pred)
        roc_auc = metrics.auc(fpr, tpr)
        plt.figure(figsize=(6,6))
        plt.title('Validation ROC')
        plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig("roc.png")

        
        torch.save(point_transformer.state_dict(),"./weights/point_transformer_epoch_{}.pth".format(epoch))





if __name__ == '__main__':
    train()

