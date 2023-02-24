#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：my_nerf 
@File    ：run_nerf.py
@IDE     ：PyCharm 
@Author  ：王子安
@Date    ：2023/2/21 9:42 
"""
import configargparse  # 这个库用来读取配置文件
import numpy as np
import time
import torch

from load import load_blender_data
from rays import get_rays_np, render
from tqdm import tqdm, trange
from einops import rearrange, reduce


def config_parser():
    parser = configargparse.ArgumentParser()  # 创建一个解析对象

    # 向其中存配置
    # 数据类型
    parser.add_argument('--dataset_type', type=str, default='blender',
                        help='目前只能加载blender数据')

    # 数据存储路径
    parser.add_argument("--datapath", type=str, default='./data/blender/chair',
                        help='数据的存储路径')

    parser.add_argument("--half_res", action='store_true',  # 只加载一半分辨率的图片，要不然爆显存了
                        help='原图是800*800的，只加载一半分辨率，400*400')

    # 给xyz坐标编码的设置 # 就是编码的指数最高是几 编码编到 multires-1 (sin(x), cos(x). sin(2x), cos(2x), ..., sin(2^9x), cos(2^9x))
    parser.add_argument("--multires", type=int, default=10,
                        help='给坐标编码编几层')
    # 同上
    parser.add_argument("--multires_views", type=int, default=4,
                        help='给视角编码编几层')

    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,  # 一轮训练中训练的光线（像素点）的数量，太大了可能会爆显存，可以调
                        help='一轮训练中训练的光线（像素点）的数量')

    return parser


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('正在使用', device)
    np.random.seed(0)

    # step 首先加载配置
    parser = config_parser()
    args = parser.parse_args()

    # step 1.加载数据 2.设定渲染范围
    if args.dataset_type == 'blender':
        # all_image[n, h, w, 4] all_poses[n, h, w] hwf[h, w, focal]
        all_images, all_poses, hwf, i_split = load_blender_data(args.datapath, args.half_res)  # 读取数据
        i_train, i_val, i_test = i_split  # 每个类别对应的下标列表 ndarray
        print('成功加载blender数据', all_images.shape, type(all_images), all_poses.shape, type(all_poses), hwf, len(i_split))
        print('train个数', type(i_train), len(i_train), 'test个数', type(i_test), len(i_test), 'val个数', type(i_val), len(i_val))

        # 设定渲染光线的边界框
        near = 2.
        far = 6.

        # 图片本来是png格式的，将其透明部分转为白色，如果是透明部分，A通道的值就是0，按下面的公式算一下最后透明部分的像素rgb通道都会被改为1，也就是白色
        # 只有透明的地方受影响
        all_images = all_images[..., :3] * all_images[..., -1:] + (1. - all_images[..., -1:])
        print('转为白色背景后的alli_images', all_images.shape, type(all_images))

    # 将图片的长和宽从float改为int
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]  # [int int numpy.float64]

    # step 设置相机内参
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])

    # step 创建NeRF网络模型

    # step 获取训练需要的数据
    rays = np.stack([get_rays_np(H, W, K, p) for p in all_poses[:, 0: 3, 0: 4]], 0)  # [N, rays_o+rays_d(2), W, H, 3] 就是每张图片的每个像素的ro和rd，三维向量 最后一维3就是具体的向量
    print('rays', rays.shape)

    all_images_rearrange = rearrange(all_images, 'n h w c -> n 1 h w c', c=3)  # 把图片改成这种形状，最后一维3就是颜色，就能和rays拼起来了
    print('all_images', all_images_rearrange.shape)

    rays_rgb = np.concatenate([rays, all_images_rearrange], 1)  # [N, ro+rd+rgb(3), H, W, 3] 把光线的原点、方向、以及这条光线对应的像素颜色结合到一起
    print('rays_rgb', rays_rgb.shape)
    rays_rgb = rearrange(rays_rgb, 'n c2 h w c -> n h w c2 c', c=3)  # 这个顺序才比较符合逻辑
    rays_rgb_train = np.stack([rays_rgb[i] for i in i_train], 0)  # 只取出训练用的数据
    print('rays_rgb_train', rays_rgb_train.shape)
    rays_rgb_train = rearrange(rays_rgb_train, 'n h w c2 c -> (n h w) c2 c')  # 因为要使用批处理，所以需要的是每个像素点的数据，而不是每张图片的数据，把rays_rgb以像素点为单位分开
    print('rays_rgb_train', rays_rgb_train.shape)
    rays_rgb_train = rays_rgb_train.astype(np.float32)
    np.random.shuffle(rays_rgb_train)  # 打乱顺序

    # # 把数据放到GPU(CPU)上
    all_images = torch.Tensor(all_images).to(device)
    all_poses = torch.Tensor(all_poses).to(device)
    rays_rgb_train = torch.Tensor(rays_rgb_train).to(device)

    print('rays_rgb_train_on_cuda', rays_rgb_train.shape)

    # 至此，已经获取到了每个像素点的光心向量和方向向量以及对应的颜色，可以开始训练了

    # 设置训练轮数
    N_iters = 200_000 + 1
    print('开始训练')

    N_rand = args.N_rand  # 获取每一轮训练中训练的光线数量
    i_batch = 0  # 用来统计训练中总共获取到的光线的数量

    # step开始训练
    for i in trange(0, 1):
        time_start = time.time()  # 统计每轮时间

        print('rays_rgb_train_in_train', rays_rgb_train.shape)

        # step 加载光线和真实的颜色
        batch = rays_rgb_train[i_batch: i_batch + N_rand]  # 取出一批光线 [N_rand, rp+rd+rgb(3), 3]
        print('batch', batch.shape)
        batch = rearrange(batch, 'n c2 c -> c2 n c', c=3)  # 把第0维和第1维换一下 [ro+rd+rgb, N_rand, 3] 这样就变成了 [每一个像素的ro, 每一个像素的rd, 每一个像素的rgb]
        batch_rays = batch[0: 2]  # 这里面存的是ro 和 rd [ro+rd, N_rand, 3]
        target_color = batch[2]  # 这里面存的是rgb [rgb, N_rand, 3]

        i_batch += N_rand
        if i_batch >= rays_rgb_train.shape[0]:
            # 如果所有数据都训练过了，那就打乱顺序再训练
            # 这个打乱的方法只能操作tensor类型
            rand_index = torch.randperm(rays_rgb_train.shape[0])
            rays_rgb_train = rays_rgb_train[rand_index]
            i_batch = 0

        # 把光线送到模型中预测
        render(batch_rays)











train()

















