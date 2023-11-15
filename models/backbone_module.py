import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

#将当前工作目录下的 "lib" 子目录添加到 Python 模块搜索路径中，以便在运行时可以导入位于该目录下的模块或文件
sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule

class Pointnet2Backbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=0, width=1, depth=2, seed_feat_dim=256):
        super().__init__()

        self.input_feature_dim = input_feature_dim

        # --------- 4 SET ABSTRACTION LAYERS ---------
        #用于对输入点云进行子采样和特征提取。通过在局部邻域内对点云进行操作，它可以捕捉局部形状和结构信息
        #mlp后面列表中三个元素分别表示输入特征维度、经过一系列全连接处理、输出特征维度
        self.sa1 = PointnetSAModuleVotes(
                npoint=2048,
                radius=0.2,
                nsample=64,
                #mlp=[input_feature_dim, 64, 64, 128],
                mlp=[input_feature_dim] + [64 * width for i in range(depth)] + [128 * width],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.4,
                nsample=32,
                #mlp=[128, 128, 128, 256],
                mlp=[128 * width] + [128 * width for i in range(depth)] + [256 * width],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=512,
                radius=0.8,
                nsample=16,
                #mlp=[256, 128, 128, 256],
                mlp=[256 * width] + [128 * width for i in range(depth)] + [256 * width],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=256,
                radius=1.2,
                nsample=16,
                #mlp=[256, 128, 128, 256],
                mlp=[256 * width] + [128 * width for i in range(depth)] + [256 * width],
                use_xyz=True,
                normalize_xyz=True
            )

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        #聚合来自较低层次的特征，以生成更高层次的特征表示
        #self.fp1 = PointnetFPModule(mlp=[256+256,256,256])
        #self.fp2 = PointnetFPModule(mlp=[256+256,256,256])
        self.fp1 = PointnetFPModule(mlp=[256 * width + 256 * width, 256 * width, 256 * width])
        self.fp2 = PointnetFPModule(mlp=[256 * width + 256 * width, 256 * width, seed_feat_dim])

   
    #将输入点云数据最后维度信息拆分成坐标信息和特征信息两部分
    def _break_up_pc(self, pc):
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, data_dict):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            data_dict: {XXX_xyz, XXX_features, XXX_inds}坐标、特征、索引信息
                XXX_xyz: float32 Tensor of shape (B,K,3)#K是每个点云中点的数量
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]#表示每个点云中点的索引
        """
        
        pointcloud = data_dict["point_clouds"] # batch, num_points, 4 (16, 40000, 4)
        batch_size = pointcloud.shape[0]
        # features: batch, 1, num_points (16, 1, 40000)
        xyz, features = self._break_up_pc(pointcloud)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        data_dict['sa1_inds'] = fps_inds
        data_dict['sa1_xyz'] = xyz
        data_dict['sa1_features'] = features

        xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        data_dict['sa2_inds'] = fps_inds
        data_dict['sa2_xyz'] = xyz
        data_dict['sa2_features'] = features

        xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        data_dict['sa3_xyz'] = xyz
        data_dict['sa3_features'] = features

        xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        data_dict['sa4_xyz'] = xyz
        data_dict['sa4_features'] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(data_dict['sa3_xyz'], data_dict['sa4_xyz'], data_dict['sa3_features'], data_dict['sa4_features'])
        features = self.fp2(data_dict['sa2_xyz'], data_dict['sa3_xyz'], data_dict['sa2_features'], features)
        data_dict['fp2_features'] = features # batch_size, feature_dim, num_seed, (16, 256, 1024)
        data_dict['fp2_xyz'] = data_dict['sa2_xyz']
        num_seed = data_dict['fp2_xyz'].shape[1]
        data_dict['fp2_inds'] = data_dict['sa1_inds'][:,0:num_seed] # indices among the entire input point clouds
        return data_dict

if __name__=='__main__':
    backbone_net = Pointnet2Backbone(input_feature_dim=3).cuda()
    print(backbone_net)
    backbone_net.eval()#将模型设置为评估模式
    #生成一个形状为 (16, 20000, 6) 的随机张量，表示一个批量大小为16，每个点云有20000个点，每个点有6个特征
    out = backbone_net(torch.rand(16,20000,6).cuda())
    for key in sorted(out.keys()):
        print(key, '\t', out[key].shape)
