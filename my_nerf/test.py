#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：my_nerf 
@File    ：test.py
@IDE     ：PyCharm 
@Author  ：王子安
@Date    ：2023/2/21 10:13 
"""

import numpy as np
from einops import rearrange
from rays import get_rays_np

a = np.arange(1, 10).reshape((3, 3))
b = np.arange(11, 20).reshape((3, 3))
c = np.arange(101, 110).reshape((3, 3))


# # np.stack
# d = np.stack((a, b, c), axis=0)
# print(d)

# # TODO 以下可以保留
# # dirs
# i, j = np.meshgrid(np.arange(10, dtype=np.float32), np.arange(10, dtype=np.float32), indexing='xy')
# # print(j)
# # print(i - 3)
# # print(-np.ones_like(i))
# dirs = np.stack([(i - 3), -(j - 3), -np.ones_like(i)], -1)
# print(dirs)
# print(dirs.shape)
#
# dirs_new = dirs[..., np.newaxis, :]
# dirs_new1 = rearrange(dirs, 'w h d -> w h 1 d', d=3)
# print('new---------------------')
#
# print(dirs_new)
# print('=======================================================')
# print(dirs.shape)
# print(dirs_new.shape)
# print(dirs_new1.shape)
#
# arr = np.arange(1, 10).reshape(3, 3)
# print(arr.shape)
# print('--------------------------------------------------------------')
# print((dirs_new1.dot(arr)).shape)
# ray = np.sum(dirs_new1 * arr, -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
# print(ray.shape)
# # ray1 = [dir.dot(arr) for dir in dirs_new1]
# ray1 = dirs_new1.dot(arr)
# print(type(ray1))
# print(len(ray1))
# print(ray1.shape)
# print('-----------------------=-=-=-============')
# rr = np.dot(dirs_new1, arr)
# print(rr.shape)
# # ray1 = np.array(ray1)
# # print(ray1.shape)
# # TODO 以上可以保留

K = np.array([
        [2, 0, 0.5 * 3],
        [0, 2, 0.5 * 3],
        [0, 0, 1]
    ])

arr = np.arange(1, 17).reshape(4, 4)

# rays_o, rays_d = get_rays_np(800, 800, K, arr[0: 3, 0: 4])
rays = np.stack(get_rays_np(3, 3, K, arr[0: 3, 0: 4]), 0)
# print(rays_o.shape)
# print(rays_d.shape)
print(rays.shape)
print(rays[0])







