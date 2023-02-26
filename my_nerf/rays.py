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

from einops import rearrange, reduce, repeat


# 首先需要一个获取相机原点到图片中每个像素的光线方向的函数 o+td 获取这个o和d
def get_rays_np(H, W, K, c2w):
    """
    获得一张图片中每一个像素对应的视角和光心
    :param H:
    :param W:
    :param K:
    :param c2w:
    :return:
    """
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


def raw2outputs(raw, )


def render_rays(ray_batch,  # 光线
                N_samples, perturb,  # 从粗光线上获取采样点需要的参数 粗采样点个数 采样点位置扰动
                model, network_query_fn,  # 计算
                N_importance,  # 从精细光线上获取采样点需要的参数 精细采样点个数
                model_fine,  # 计算
                white_bkgd,  # 是否使用了白色背景
                raw_noise_std,  # TODO 还没用到 不知道干嘛的
                pytest=False
                ):
    # 取出数据
    # ray_batch [N_rand, 3+3+1+1+3] rays_o, rays_d, near, far, view_dirs
    N_rays = ray_batch.shape[0]  # 光线的数量
    rays_o = ray_batch[:, 0: 3]  # 取出光心 [N_rand, 3]
    rays_d = ray_batch[:, 3: 6]  # 取出方向 [N_rand, 3]
    view_dirs = ray_batch[:, -3:]  # 取出视角（其实就是规则化的方向） [N_rand, 3]
    near = ray_batch[:, 6: 7]  # [N_rand, 1]
    far = ray_batch[:, 7: 8]  # [N_rand, 1]

    # 放置采样点
    t_vals = torch.linspace(0., 1., steps=N_samples)  # o+td中的t
    print('t_vals', t_vals.shape)
    z_vals = 1 / ((1. / near) * (1. - t_vals) + (1. / far) * t_vals)  # [N_N_rand, N_samples] 范围也是2-6 但是是非线性变化的 sample linearly in inverse depth 看函数图像，这样采样的话距离原点越近采样点越密集，应该距离原点越近采样效果就越好，可能相当于给距离近的点加权重了
    print('z_vals', z_vals.shape)

    # 给采样点添加扰动 让每个采样点的位置在一个小邻域内浮动
    if perturb > 0.:
        # 设置扰动
        mids = 0.5 * (z_vals[:, : -1] + z_vals[:, 1:])  # [N_rand, N_samples-1] 每两个点的中间位置
        # 每个点在upper和lower之间扰动
        upper = torch.cat([mids, z_vals[:, -1:]], -1)  # [N_rand, M_samples] 每个点扰动的上界
        lower = torch.cat([z_vals[:, 0: 1]], -1)  # [N_rand, M_samples] 每个点扰动的下界
        t_rand = torch.rand(z_vals.shape)  # [N_rand, M_samples] 返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义

        # 应用扰动
        z_vals = lower + t_rand * (upper - lower)

    # rays_o [N_rays ,3]            ->[N_rand, 1, 3]
    # rays_d[N_rays, 3]             ->[N_rand, 1, 3]
    # z_vals[N_rays, N_samples]     ->[N_rand, N_samples, 1]
    # 采样点坐标是o+td t就是z_vals
    rays_o_expand = rearrange(rays_o, 'n d -> n 1 d', d=3)
    rays_d_expand = rearrange(rays_d, 'n d -> n 1 d', d=3)
    z_vals_expand = rearrange(z_vals, 'n n2 -> n n2 1')

    # 其实这三个tensor已经可以直接运算了，因为tensor星乘*可以自动广播，但为了看的更直观，手动广播一下
    rays_o_expand = repeat(rays_o_expand, 'n n2 d -> n (repeat n2) d', repeat=N_samples, d=3)  # [N_rand, N_samples, 3]
    rays_d_expand = repeat(rays_d_expand, 'n n2 d -> n (repeat n2) d', repeat=N_samples, d=3)  # [N_rand, N_samples, 3]
    z_vals_expand = repeat(z_vals_expand, 'n n2 d -> n n2 (repeat d)', repeat=3)  # [N_rand, N_samples, 3]

    pts = rays_o_expand + z_vals_expand * rays_d_expand  # [N_rand, N_samples, 3]
    # 到此位置已经获得采样光线了，该放到网络里预测了

    # 获得渲染出来的颜色
    # 把光线和视角放到网络里，获取预测的结果[rgb, a]
    raw = network_query_fn(pts, view_dirs, model)
    print(raw.shape)
    # 用体渲染方程获得颜色


def batchify_rays(rays, chunk=32 * 32 * 4 * 8, **kwargs):
    """
    负责把光线分批
    :param rays: 光线
    :param chunk: 一批的大小
    :param kwargs: 参数
    :return:
    """
    all_ret = dict()
    for i in range(0, rays.shape[0], chunk):
        print('正在处理第', i, '批光线')
        render_rays(rays[i: i + chunk], **kwargs)


# 调用这个函数的时候，near 和 far在字典里，但是定义这个函数的时候，不在字典里，这样可以方便本函数调用，省的再从字典里拿，这个写法不错，可以参考
def render(H, W, K, chunk=32 * 32 * 4 * 8, rays=None, near=0., far=1., **kwargs):
    """
    负责整个的渲染流程
    :param K:
    :param W:
    :param H:
    :param rays:
    :param near:
    :param far:
    :param chunk:
    :param kwargs:
    :return:
    """
    rays_o, rays_d = rays  # 获取光线 [N_rand, 3]

    # 获取视角 并转为float
    view_dirs = rays_d
    view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)  # torch.norm是求范数 这个式子就是把view_dirs正则化到0-1之间 并且保持维数不变 让每个像素都有对应的视角
    view_dirs = view_dirs.float()  # [N_rand, 3]

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
    batchify_rays(rays=rays_full, chunk=chunk, **kwargs)












