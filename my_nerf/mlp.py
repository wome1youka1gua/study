#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：my_nerf 
@File    ：mlp.py
@IDE     ：PyCharm 
@Author  ：王子安
@Date    ：2023/2/20 21:35 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# 编码器类
class Embedder:
    def __init__(self, **kwargs):
        self.embed_fns = None
        self.out_dim = None
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        """
        用来创造一个编码函数
        :return:
        """
        embed_fns = []  # 用来存编码函数 sin cos
        d = self.kwargs['input_dims']  # 输入的坐标的维数 无论是xyz坐标或者视角坐标 都是3
        out_dim = 0  # 输出的向量的维数

        # 是否要包括输入
        if self.kwargs['included_input']:
            embed_fns.append(lambda x: x)
            out_dim += d  # 统计输出的向量的维数

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)  # torch.tensor [2.^0, 2.^1, ... ,2.^(L-1)]
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))  # 编码函数列表 sin(x * 2^n) cos(x * 2^n)
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


# step 获取编码器和编码后的向量维度
def get_embedder(multires):
    # 编码参数
    embed_kwargs = {
        'included_input': True,  # 是否包含原始坐标，看一眼论文的网络结构
        'input_dims': 3,  # 要编码的坐标的维度
        'max_freq_log2': multires - 1,  # 就是编码的指数最高是几 编码编到(sin(x), cos(x). sin(2x), cos(2x), ..., sin(2^9x), cos(2^9x))
        'num_freqs': multires,  # 作为torch.linspace的步长
        'periodic_fns': [torch.sin, torch.cos]
    }
    embedder_obj = Embedder(**embed_kwargs)
    # 创建一个编码器
    embed = lambda x, eo=embedder_obj: eo.embed(x)

    # 编码器的输入：坐标或视角，也就是输入x，x是这样的tensor a = torch.tensor([[1, 2, 3], [4, 5, 6]]) n个坐标 [n, 3]
    # 编码器的输出：编码完也是tensor 坐标[n, 63]  视角[n, 27]
    return embed, embedder_obj.out_dim  # 返回编码器和编码后的坐标/视角的维数


# step 创建NeRF网络 包括创建网络(init) 前向传播(forward) 加载权重(load_weights_from_keras)
class NeRFNetwork(nn.Module):
    # D是预测密度的网络的隐藏层数量 W是每一层的神经元数量 input_ch是输入的维数 skip是第几层需要额外再把原始输入加进来
    def __init__(self, D=8, W=256, input_ch=3, input_ch_view=3, skip=[4]):
        """
        设置网络层数，每层神经元数，输入向量的维数
        :param D:
        :param W:
        :param input_ch:
        :param input_ch_view:
        :param skip:
        """
        super(NeRFNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_view = input_ch_view
        self.skip = skip

        # step 开始创建MLP
        # 首先创建输入层和前八个隐藏层
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, self.W)] + [nn.Linear(self.W, self.W) if i not in self.skip else nn.Linear(self.W + self.input_ch, self.W) for i in range(self.D - 1)]
        )
        # 这层是添加视角的那层
        self.view_linear = nn.Linear(self.W + self.input_ch_view, self.W // 2)
        # 在8个隐藏层后直接预测密度的层
        self.sigma_linear = nn.Linear(self.W, 1)

        # 8个隐藏层后继续处理，以获取颜色
        self.feature_linear = nn.Linear(W, W)  # 第一层是一个普通的层，继续处理一下
        self.rgb_linear = nn.Linear(self.W // 2, 3)  # 预测颜色的输出层 接上面添加视角的那层

        self.init_parameters()

    def forward(self, x):
        """
        前向传播
        :param x: 像素坐标和视角
        :return:
        """
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_view], dim=0)  # TODO 我觉得这个x应该就是一个一维张量，所以dim为0也没问题，原来的代码是-1

        hide_output = input_pts
        for i, linear in enumerate(self.pts_linears):
            hide_output = linear(hide_output)
            hide_output = F.relu(hide_output)

            if i in self.skip:
                hide_output = torch.cat([input_pts, hide_output], dim=0)

        # 计算密度
        sigma = self.sigma_linear(hide_output)

        # 计算颜色
        feature = self.feature_linear(hide_output)
        # 把视角也送进网络
        hide_output = torch.cat([feature, input_views], -1)
        hide_output = self.view_linear(hide_output)
        rgb = self.rgb_linear(hide_output)  # 计算颜色

        # 把颜色和密度拼起来作为输出
        outputs = torch.cat([rgb, sigma], -1)

        return outputs

    # 初始化权重
    def init_parameters(self):
        torch.manual_seed(1)  # 随机数初始化种子
        for i, linear in enumerate(self.pts_linears):
            torch.nn.init.normal_(linear.weight, mean=0, std=1)  # 表示生成的随机数用来替换这个层中的.weight的原始数据
        torch.nn.init.normal_(self.view_linear.weight, mean=0, std=1)
        torch.nn.init.normal_(self.sigma_linear.weight, mean=0, std=1)
        torch.nn.init.normal_(self.feature_linear.weight, mean=0, std=1)
        torch.nn.init.normal_(self.rgb_linear.weight, mean=0, std=1)


def create_nerf(args):
    # step 获取坐标的编码函数 以及 编码后的坐标的维度
    embed_fn, input_ch = get_embedder(args.multires)

    # step 获取视角的编码函数 以及 编码后的视角的维度
    embed_dir_fn, input_ch_views = get_embedder(args.multires_views)

    # step 初始化mlp网络 粗网络和精细网络
    grad_vars = list()  # 用来存储两个
    skip = [4]

    # step 创建mlp网络 需要创建一个粗网络和一个精细网络 以及获取网络中的权重和偏置
    model = NeRFNetwork()  # 粗网络
    grad_vars = list(model.parameters())  # 获取网络中可训练的参数 weight和bias
    model_fine = NeRFNetwork()  # 精细网络 实际上和粗网络长得一样
    grad_vars.append(model_fine.parameters())

    # step 创建一个查询网络，输入为



# # 测试代码
# em, dim = get_embedder(4)
# a = torch.tensor([[1, 2, 3], [4, 5, 6]])
# b = em(a)
# print(a)
# print(b)
# print(b.shape)
# print(type(b))
# print(type(b[0]))
# print(len(b))
# print(dim)




















