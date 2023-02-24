#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：my_nerf 
@File    ：load.py
@IDE     ：PyCharm 
@Author  ：王子安
@Date    ：2023/2/20 21:42 
"""
import os
import json

import cv2
import imageio
import numpy as np


# step 加载blender数据
def load_blender_data(base_dir, half_res=True):
    """
    加载blender数据

    数据以json格式存储，这样读取起来比较方便，因为有对应的库
    :param half_res: 是否只用一半的分辨率训练
    :param base_dir:数据所在文件夹
    :return:
    """
    splits = ['train', 'test', 'val']
    metas = {}

    # 把对应类别的数据存到字典里
    for single_split in splits:
        with open('{}/transforms_{}.json'.format(base_dir, single_split), 'r') as fp:
            metas[single_split] = json.load(fp)

    all_images = list()  # 用来存所有的图片
    all_poses = list()  # TODO 用来存所有的
    count = [0]

    # 分别从metas里面解析每个类别的数据
    for single_split in splits:
        meta = metas[single_split]
        images = list()  # 用来存储这个类别中的所有图片
        poses = list()  # TODO 用来存储这个类别中的所有

        for frame in meta['frames']:  # frame是dict meta是list
            picture_path = '{}/{}.png'.format(base_dir, frame['file_path'])  # 图片路径

            picture_array = imageio.v2.imread(picture_path)  # 将图片读取为numpy数组 保留RGBA
            images.append(picture_array)
            # print('format picture_array', type(picture_array), picture_array.shape)  # TODO 输出变量样式 image

            transform_matrix = np.array(frame['transform_matrix'])
            poses.append(transform_matrix)
            # print('format transform_matrix', type(transform_matrix), transform_matrix.shape)  # TODO 输出变量样式 transform_matrix

        # 将数据规制化
        images = (np.array(images) / 255.).astype(np.float32)  # 把图片从0-255的int转为0-1的float32 [n, h, w, 4] 每一张图片是 [h, w, 4] n张图片
        poses = (np.array(poses)).astype(np.float32)  # 同样转换格式 [n, 4, 4] n个4*4的变换矩阵

        all_images.append(images)  # 把这个类别的images list放到总的list中
        all_poses.append(poses)  # 把这个类别的poses list放到总的list中
        count.append(count[-1] + images.shape[0])  # 用来统计每种类别中的数量

    i_spilt = [np.arange(count[i], count[i + 1]) for i in range(3)] # 训练集、测试集、验证集的序号列表[array([1, 2, 3,...,m]), array([m, m + 1, ..., n]), array([n, n+ 1, ..., l])]
    # 这样就可以把不同种类的图片合到一个大列表里了
    final_all_images = np.concatenate(all_images, 0)
    final_all_poses = np.concatenate(all_poses, 0)
    # print('format final_all_pictures', '大列表类别', type(final_all_images), 'shape', final_all_images.shape, '里面每个元素的类别', type(final_all_images[0]))
    # print('format final_all_poses', '大列表类别', type(final_all_poses), 'shape', final_all_poses.shape, '里面每个元素的类别', type(final_all_poses[0]))

    H, W = final_all_images[0].shape[0: 2]  # 获取图片的高和宽
    camera_angle_x = float(meta['camera_angle_x'])  # 相机的视角范围 水平方向
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)  # 计算相机焦距
    picture_info = [H, W, focal]

    # print(final_all_poses[:, :3, :4].shape)
    # print(final_all_poses[..., :3, :4].shape)

    # 节省性能，把图片缩成一半
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        final_all_images_half_res = np.zeros((final_all_images.shape[0], H, W, 4))  # [N, H_half, W_half, 4]
        for i, image in enumerate(final_all_images_half_res):
            final_all_images_half_res[i] = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
        final_all_images = final_all_images_half_res
        picture_info = [H, W, focal]

    return final_all_images, final_all_poses, picture_info, i_spilt


# load_blender_data(r'D:\PythonProjects\my_nerf\data\blender\chair')






















