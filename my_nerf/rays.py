#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：my_nerf 
@File    ：rays.py
@IDE     ：PyCharm 
@Author  ：王子安
@Date    ：2023/2/22 16:46 
"""

import numpy as np
import torch

from einops import rearrange


# 首先需要一个获取相机原点到图片中每个像素的光线方向的函数 o+td 获取这个o和d
def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)  # (W, H, 3) 里面的每一个视角向量就是(x, -y, -1) 因为相机成像其实是小孔成像原理，所以上下是颠倒的，所以y轴要取反
    # 要将视角转到世界坐标 就要和c2w相乘 就要把矩阵形状转匹配
    dirs = rearrange(dirs, 'w h d -> w h 1 d', d=3)  # (w, h, 1, 3)
    rotate_matrix = c2w[0: 3, 0: 3]  # 从外参矩阵中获取旋转向量
    # rays_d = np.dot(dirs, rotate_matrix)  # 用旋转矩阵将视角转到世界坐标系
    rays_d = np.sum(dirs * rotate_matrix, -1)  # TODO 这为什么是星乘？
    translate_matrix = c2w[:3, -1]  # 从外参矩阵中获取平移向量
    rays_o = np.broadcast_to(translate_matrix, np.shape(rays_d))  # 外参矩阵中的平移向量就相当于相机原点在世界坐标系中的坐标，所以rays_o就是采样光线的发射点在世界坐标系中的坐标

    return rays_o, rays_d


def render(rays, near, far):
    rays_o, rays_d = rays  # 获取光线 [N_rand, 3]
    print('rays_o_in_render', rays_o.shape)
    print('rays_d_in_render', rays_d.shape)

    # 获取视角 并转为float [N_rand, 3]
    view_dirs = rays_d
    view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)  # torch.norm是求范数 这个式子就是把view_dirs正则化到0-1之间 并且保持维数不变 让每个像素都有对应的视角
    view_dirs = view_dirs.float()  # [N_rand, 3]
    print(view_dirs.shape)

    # 创建光线
    # 把rays_o 和 rays_d 也转为float
    rays_o = rays_o.float()  # [N_rand, 3]
    rays_d = rays_d.float()  # [N_rand, 3]

    # 获取远近距离限制
    near = near * torch.ones_like(rays_d[:, : 1])  # [N_rand, 1]
    far = far * torch.ones_like(rays_d[:, : 1])  # [N_rand, 1]

    # 拼接成完整的光线
    rays_full = torch.cat([rays_o, rays_d, near, far, view_dirs], -1)  # 拼接到一起 [N_rand, 3+3+1+1+3]

    # 开始渲染并格式化













