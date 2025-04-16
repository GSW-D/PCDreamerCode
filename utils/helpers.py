# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 18:34:19
# @Email:  cshzxie@gmail.com

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
import random
from models.model_utils import fps_subsample
import torch.nn.functional as F
import cv2
import re


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.ConvTranspose2d or \
            type(m) == torch.nn.Conv3d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def count_parameters(network):
    return sum(p.numel() for p in network.parameters())


def get_ptcloud_img(ptcloud):
    fig = plt.figure(figsize=(8, 8))

    x, z, y = ptcloud.transpose(1, 0)
    ax = fig.gca(projection=Axes3D.name, adjustable='box')
    ax.axis('off')
    ax.axis('scaled')
    ax.view_init(30, 45)

    max, min = np.max(ptcloud), np.min(ptcloud)
    ax.set_xbound(min, max)
    ax.set_ybound(min, max)
    ax.set_zbound(min, max)
    ax.scatter(x, y, z, zdir='z', c=x, cmap='jet')

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img


def save_depth_img(depth, view_num, save_path):
    """

    :param depth: ndarray of depth image [B*num_views, RESOLUTION, RESOLUTION]
    :param view_num: num of views
    :param save_path:
    :return:
    """
    imgs = depth[:view_num, :, :]
    for i in range(view_num):
        img = imgs[i]
        mask = np.argwhere(img > 0)
        print(img.shape)
        img[mask] = (img[mask] - img[mask].min()) / (img[mask].max() - img[mask].min() + 1e-4) * 255.0
        # img[mask] = 1-img[mask]
        # img = np.expand_dims(imgs[i], -1)
        # mask_idx = np.argwhere(img == 0)[:, :2]
        # img_idx = np.argwhere(img != 0)[:, :2]
        # min_xy = np.min(img_idx, axis=0)
        # max_xy = np.max(img_idx, axis=0)
        # cen = (min_xy + max_xy) // 2
        # crop_size = np.max(max_xy - min_xy) // 2 + 1
        # start_xy = np.clip(cen - crop_size, 0, img.shape[0] - 1)
        # end_xy = np.clip(cen + crop_size, 0, img.shape[0] - 1)
        # # result = cv2.applyColorMap(cv2.convertScaleAbs(img, alpha=0.5), cv2.COLORMAP_JET)
        # # result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
        # # result[idx[:, 0], idx[:, 1], 3] = 0.0
        # img[mask_idx[:, 0], mask_idx[:, 1], :] = 255.0
        # result = img[start_xy[0]:end_xy[0], start_xy[1]:end_xy[1], :]
        # if result is None:
        #     print(save_path)
        img = cv2.dilate(np.uint8(img), kernel=np.ones((5, 5), dtype=np.uint8))
        cv2.imwrite(save_path + f'_view{i}.png', img)


def seprate_point_cloud(xyz,
                        num_points,
                        crop,
                        fixed_points=None,
                        padding_zeros=False):
    '''
     seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    '''
    _, n, c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None

    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop, list):
            num_crop = random.randint(crop[0], crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:
            center = F.normalize(torch.randn(1, 1, 3), p=2, dim=-1).cuda()
        else:
            if isinstance(fixed_points, list):
                fixed_point = random.sample(fixed_points, 1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1, 1, 3).cuda()

        distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1),
                                     p=2,
                                     dim=-1)  # 1 1 2048

        idx = torch.argsort(distance_matrix, dim=-1,
                            descending=False)[0, 0]  # 2048

        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] = input_data[0, idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0,
            idx[num_crop:]].unsqueeze(0)  # 1 N 3

        crop_data = points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop, list):
            INPUT.append(fps_subsample(input_data, 2048))
            CROP.append(fps_subsample(crop_data, 2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT, dim=0)  # B N 3
    crop_data = torch.cat(CROP, dim=0)  # B M 3

    return input_data.contiguous(), crop_data.contiguous()


def pc_normalize(pc):
    """

    :param pc: [B, N, 3]
    :return: pc: [B, N, 3], centroid: [B, 3], m: [B,]
    """
    B, _, _ = pc.shape
    centroid = torch.mean(pc, dim=1)
    pc = pc - centroid.view(B, 1, 3)
    m, _ = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=-1)), dim=-1)
    pc = pc / m.view(B, 1, 1)
    return pc, centroid, m


def pc_normalize2(pc):
    """

    :param pc: [B, N, 3]
    :return: pc: [B, N, 3], centroid: [B, 3], m: [B,]
    """
    B, _, _ = pc.shape
    centroid = torch.mean(pc, dim=1)
    pc = pc - centroid.view(B, 1, 3)
    m, _ = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=-1)), dim=-1)
    pc = pc / (m.view(B, 1, 1) * 2)
    return pc, centroid, m


def upsample_points(ptcloud, n_points):
    curr = ptcloud.shape[0]
    need = n_points - curr

    if need < 0:
        return ptcloud

    while curr <= need:
        ptcloud = np.tile(ptcloud, (2, 1))
        need -= curr
        curr *= 2

    choice = np.random.permutation(need)
    ptcloud = np.concatenate((ptcloud, ptcloud[choice]))

    return ptcloud


def read_pfm(path):
    """Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale


def write_pfm(path, image, scale=1):
    """Write pfm file.

    Args:
        path (str): pathto file
        image (array): data
        scale (int, optional): Scale. Defaults to 1.
    """

    with open(path, "wb") as file:
        color = None

        if image.dtype.name != "float32":
            raise Exception("Image dtype must be float32.")

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif (
            len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
        ):  # greyscale
            color = False
        else:
            raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

        file.write("PF\n" if color else "Pf\n".encode())
        file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == "<" or endian == "=" and sys.byteorder == "little":
            scale = -scale

        file.write("%f\n".encode() % scale)

        image.tofile(file)