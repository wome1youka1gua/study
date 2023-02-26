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
import torch.nn.functional as F

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


def sample_pdf(z_vals_mid, transmittance, N_samples,  det=False, pytest=False):
    """

    :param z_vals_mid:
    :param transmittance: [N_rand, N_samples]
    :param N_samples:
    :param det:
    :param pytest:
    :return:
    """
    transmittance = transmittance + 1e-5   # prevent nans

    pdf = transmittance / torch.sum(transmittance, -1)  # [N_rand, N_samples] 求transmittance的概率密度函数
    print('pdf.shape', pdf.shape)
    cdf = torch.cumsum(pdf, -1)  # [N_rand, N_samples] 分布函数 求导就是transmittance变化率
    zeros_for_cdf = torch.zeros_like(cdf[:, 0: 1])
    cdf = torch.cat([zeros_for_cdf, cdf], -1)  # [N_rand, N_samples + 1] 在分布函数前面补个0，分布函数第一位变化率是0，所以得补0

    if not det:
        u = torch.rand(list(cdf.shape[0: 1]) + [N_samples])  # [N_rand, N_samples]
    u = u.contiguous()  # 让u的内存连续




def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """

    :param raw: [N_rand, N_samples, 4]
    :param z_vals: [N_rand, N_samples] 范围是2到6
    :param rays_d: [N_rand, 3]
    :param raw_noise_std:
    :param white_bkgd:
    :param pytest:
    :return:
    """
    # 通过密度求透明度 alpha_i = 1 - e^-(sigma_i * delta_i)
    raw2alpha = lambda sigma, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(sigma) * dists)  # 为了避免sigma预测出负数 用relu处理一下

    # 两点之间的距离
    dists = z_vals[:, 1:] - z_vals[:, :-1]  # [N_rand, N_samples - 1]

    dist_infi = torch.Tensor([1e10])  # 这个项目里无限远的距离是1^10
    dist_infi = rearrange(dist_infi, 'n -> 1 n')
    dist_infi = repeat(dist_infi, 'n n1 -> (repeat n) n1', repeat=dists.shape[0])  # [N_rand, 1]

    dists = torch.cat([dists, dist_infi], -1)  # [N_rand, N_samples]

    # 采样的时候没有乘光线方向的长度，而渲染的时候乘了 应该是因为 渲染出来的结果说是人眼看到的，实际上就是相机看到的，就像ue和blender里一样，所以渲染的时候为了模拟看起来的情况，其实就是要模拟相机看到的情况，而相机的视角是一个等腰三角形，腰长肯定比中垂线长
    # 所以要乘一下，才能转化为真实世界中的距离。采样的时候没乘，应该是因为不需要，采样的时候只需要知道每一个点的颜色就行，而不是在相机中一个方向的颜色，实际上采样里的z_vals和渲染的z_vals的点不是一组点，关系不大，采样的目的就是要得出这一条光线上的所有的点
    # 的颜色，而渲染的时候只需要这条光线上的一部分点的颜色
    rays_d_for_norm = rearrange(rays_d, 'n d -> n 1 d', d=3)  # [N_rand, 1, 3] 不扩一维下面求范数的时候会少一维，不好乘了，不管先扩还是后扩都得扩
    rays_d_norm = torch.norm(rays_d_for_norm, dim=-1)  # [N_rand, 1]
    dists = dists * rays_d_norm  # [N_rand, N_samples] 每个点到下一个点的距离 delta_i

    # 从raw中取出颜色
    rgb = torch.sigmoid(raw[:, :, 0: 3])  # [N_rand, N_samples, 3] TODO 这使用sigmoid是为了激活还是规制？ 我感觉好像是激活 因为反向传播的时候算误差用的是已经归一化的颜色作为目标颜色 这里应该不需要在归一了 但是为什么还需要激活呢？

    # 给要渲染的点的密度添加扰动
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[:, :, 3].shape) * raw_noise_std  # [N_rand, N_samples]
        print('noise.shape', noise.shape)

    # 计算alpha
    sigma = raw[:, :, 3] + noise  # [N_rand, N_samples]
    alpha = raw2alpha(sigma=sigma, dists=dists)  # [N_rand, N_samples]
    print('alpha.shape', alpha.shape)

    # 计算transmittance
    # torch.cumprod [1, (1 - a1), (1 - a1)(1 - a2), ... , (1 - a1)(1 - a2)...(1 - an-1)]
    ones_for_cumprod = torch.ones((alpha.shape[0], 1))  # [N_rand, 1]
    alpha_for_cumprod = torch.cat([ones_for_cumprod, 1. - alpha + 1e-10], -1)  # [N_rand, N_samples + 1] [N_rand 个 [1, a1, a2, ..., aN_samples]] 最后加一个1e-10是为了防止乘积为0，所以加一个比较小的正数，不会太影响结果
    alpha_cumprod = torch.cumprod(alpha_for_cumprod, -1)  # [N_rand, N_samples + 1]
    transmittance = alpha * alpha_cumprod[:, :-1]  # [N_rand, N_samples] [a1, a2(1 - a1), a3(1 - a1)(1 - a2), ... , an(1 - a1)(1 - a2)...(1 - aN_samples-1)]

    # 根据MLP预测出来的颜色和transmittance就可以用体渲染公式算出来相机看到的颜色了
    transmittance_rearrange = rearrange(transmittance, 'n n1 -> n n1 1')
    transmittance_expand = repeat(transmittance_rearrange, 'n n1 d -> n n1 (repeat d)', repeat=3, d=1)  # [N_rand, N_samples, 3]
    rgb_map = torch.sum(rgb * transmittance_expand, -2)  # [N_rand, 3] 就是把每一条光线上面采样点的颜色计算结果加一起，算出了这个方向看过去的颜色 TODO 这个好像也可以用einops替代 AI葵(2) 1:28:54

    # 算相机看过去的这个点和相机之间的距离应该是多少 计算方法和颜色一样 只不过把颜色c换成距离z
    depth_map = torch.sum(z_vals * transmittance, -1)  # [N_rand]

    # TODO 这个好像没啥用 不知道是啥 不过估计这个在输出预测的模型obj的时候才需要
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(transmittance, -1))

    # 告诉你这个点是不是透明的
    acc_map = torch.sum(transmittance, -1)  # [N_rand]

    # 把最后渲染的图片弄成白色背景
    if white_bkgd:
        acc_map_expand = rearrange(acc_map, 'n -> n 1')
        rgb_map = rgb_map + (1. - acc_map_expand)

    return rgb_map, disp_map, acc_map, transmittance, depth_map


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
    z_vals = 1 / ((1. / near) * (1. - t_vals) + (1. / far) * t_vals)  # [N_rand, N_samples] 是2-6 但是是非线性变化的 sample linearly in inverse depth 看函数图像，这样采样的话距离原点越近采样点越密集，应该距离原点越近采样效果就越好，可能相当于给距离近的点加权重了

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
    # 把所有点的坐标和视角放到网络里，获取预测的结果[rgb, a]
    raw = network_query_fn(pts, view_dirs, model)  # [N_rand, N_samples, 4]
    # 用体渲染方程获得颜色
    rgb_map, disp_map, acc_map, transmittance, depth_map = raw2outputs(raw=raw, z_vals=z_vals, rays_d=rays_d, raw_noise_std=raw_noise_std, white_bkgd=white_bkgd, pytest=pytest)

    # TODO 先跳过吧 这有点难
    # 如果N_importance大于0，就对weight变化率更大（相当于密度更大）的那个区域进行采样
    if N_importance > 0.:
        rgb_map0, disp_map0, acc_map0 = rgb_map, disp_map, acc_map

        z_vals_mids = 0.5 * (z_vals[:, : -1] + z_vals[:, 1:])  # [N_rand, N_samples-1] 每两个点的中间位置

    # TODO 2023-2-26 15:06 该ret了


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












