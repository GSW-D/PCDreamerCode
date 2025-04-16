import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
# from monai.metrics import *
import cv2
import open3d as o3d
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from models.model_utils import fps_subsample_v2


def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    def lr_func(epoch):
        if epoch <= start_epoch:
            return 1.0
        elif epoch <= end_epoch:
            total = end_epoch - start_epoch
            delta = epoch - start_epoch
            frac = delta / total
            return (1 - frac) * 1.0 + frac * (end_lr / start_lr)
        else:
            return end_lr / start_lr

    return LambdaLR(optimizer, lr_lambda=lr_func)


def ChamferDistance(p1, p2):
    """
    Calculate Chamfer Distance between two point sets
    Input:
        p1: [bn, N, D]
        p2: [bn, M, D]
    Return:
        CD: sum of Chamfer Distance of two point sets
    """
    diff = p1[:, :, None, :] - p2[:, None, :, :]
    dist = torch.sum(diff * diff, dim=3)
    dist1 = dist
    dist2 = torch.transpose(dist, 1, 2)

    dist_min1, _ = torch.min(dist1, dim=2)
    dist_min2, _ = torch.min(dist2, dim=2)

    CD = (torch.sum(dist_min1) / (p1.shape[1]) + torch.sum(dist_min2) / (p2.shape[1])) / (p1.shape[0])
    return CD


def OneWayChamferDistance(p1, p2):
    """
    Calculate Chamfer Distance from p1 to p2
    Input:
        p1: [bn, N, D]
        p2: [bn, M, D]
    Return:
        OWCD: sum of Chamfer Distance of two point sets
    """
    diff = p1[:, :, None, :] - p2[:, None, :, :]
    dist = torch.sum(diff * diff, dim=3)
    dist1 = dist
    dist2 = torch.transpose(dist, 1, 2)

    dist_min1, _ = torch.min(dist1, dim=2)
    dist_min2, _ = torch.min(dist2, dim=2)

    OWCD = torch.sum(dist_min1) / (p1.shape[1]) / (p1.shape[0])
    return OWCD


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def get_mask_from_norm(normal_map):
    mask_ = torch.norm(normal_map, dim=1, keepdim=True) < 0.5
    mask = torch.norm(normal_map, p=2, dim=1, keepdim=True) >= 0.5
    return mask, mask_


def to_visual_image(depth_map, normal_map, rejection=False, masked=True, threshold=(-2, 2, -1, 1)):
    mask, mask_ = get_mask_from_norm(normal_map)
    d_th_bottom, d_th_upper, d_th = threshold[0], threshold[1], threshold[1] - threshold[0]
    n_th_bottom, n_th_upper, n_th = threshold[2], threshold[3], threshold[3] - threshold[2]
    if rejection:
        # 计算投影
        depth_fig = (depth_map - depth_map[mask].min()) / (depth_map[mask].max() - depth_map[mask].min()) * 255
        normal_fig = (normal_map.clip(-1, 1) + 1) / 2 * 255
    else:
        # 截断可视化
        depth_fig = (depth_map.clip(d_th_bottom, d_th_upper) - d_th_bottom) / d_th * 255
        normal_fig = (normal_map.clip(n_th_bottom, n_th_upper) - n_th_bottom) / n_th * 255
    if masked:
        depth_fig = depth_fig.masked_fill(mask_, 0)
        normal_fig = normal_fig.masked_fill(mask_.repeat(1, 3, 1, 1), 0)

    # cv2.imshow('normal', np.uint8(normal_fig[0].permute(1, 2, 0).cpu().numpy()))
    # cv2.imshow('depth', np.uint8(depth_fig[0].permute(1, 2, 0).cpu().numpy()))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return depth_fig, normal_fig


def to_visual_image_single(map, mask=None, threshold=(-2, 2)):
    threshold_bottom, threshold_upper, width = threshold[0], threshold[1], threshold[1] - threshold[0]
    figure = (map.clip(threshold_bottom, threshold_upper) - threshold_bottom) / width * 255
    if mask is not None:
        figure = figure * mask
    return figure


def depth_to_pointcloud(depth_map, fx=16, fy=16, cx=256, cy=256, keep_dim=False):
    """
    将深度图转换为点云。
    :param depth_map: 深度图，形状为Bx1xHxW
    :param fx, fy: 相机的焦距
    :param cx, cy: 相机的主点坐标
    :param keep_dim: 是否合并HxW
    :return: 点云，形状为Bx3x(HxW)
    """
    B, _, H, W = depth_map.shape
    device = depth_map.device
    # 生成网格，表示每个像素的(u, v)坐标
    u_coord = torch.arange(W, device=device).view(1, 1, 1, W).repeat(B, 1, H, 1)
    v_coord = torch.arange(H, device=device).view(1, 1, H, 1).repeat(B, 1, 1, W)
    # 计算x和y坐标
    x = (u_coord - cx) / fx
    y = (v_coord - cy) / fy
    z = depth_map
    # 组合x, y, z坐标形成点云
    pointcloud = torch.cat((x, y, z), dim=1)
    if not keep_dim:
        pointcloud = pointcloud.view(B, 3, -1).permute(0, 2, 1)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pointcloud[0].cpu().numpy())
    # o3d.visualization.draw_geometries([pcd])

    return pointcloud


def geometrymap_to_pointcloud(geometrymap, fx=16, fy=16, cx=256, cy=256, keep_dim=False):
    """
    将深度图转换为点云。
    :param geometrymap: 形状为Bx4xHxW
    :param fx, fy: 相机的焦距
    :param cx, cy: 相机的主点坐标
    :param keep_dim: 是否合并HxW
    :return: 点云，形状为Bx6x(HxW)
    """
    B, _, H, W = geometrymap.shape
    device = geometrymap.device
    # 生成网格，表示每个像素的(u, v)坐标
    u_coord = torch.arange(W, device=device).view(1, 1, 1, W).repeat(B, 1, H, 1)
    v_coord = torch.arange(H, device=device).view(1, 1, H, 1).repeat(B, 1, 1, W)
    # 计算x和y坐标
    x = (u_coord - cx) / fx
    y = (v_coord - cy) / fy
    z = geometrymap
    # 组合x, y, z坐标形成点云
    pointcloud = torch.cat((x, y, z), dim=1)
    if not keep_dim:
        pointcloud = pointcloud.view(B, 6, -1).permute(0, 2, 1)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pointcloud[0].cpu().numpy())
    # o3d.visualization.draw_geometries([pcd])

    return pointcloud


def geometrymap_to_pointcloud_v2(geometrymap, mask, fx=16, fy=16, cx=256, cy=256, keep_dim=False):
    """
    将深度图转换为点云。
    :param geometrymap: 形状为Bx4xHxW
    :param fx, fy: 相机的焦距
    :param cx, cy: 相机的主点坐标
    :param keep_dim: 是否合并HxW
    :return: 点云，形状为Bx6x(HxW)
    """
    B, _, H, W = geometrymap.shape
    device = geometrymap.device
    valid_ids = (mask > 0)
    # 生成网格，表示每个像素的(u, v)坐标
    u_coord = torch.arange(W, device=device).view(1, 1, 1, W).repeat(B, 1, H, 1)
    v_coord = torch.arange(H, device=device).view(1, 1, H, 1).repeat(B, 1, 1, W)
    # 计算x和y坐标
    x = (u_coord - cx) / fx
    y = (v_coord - cy) / fy
    z = geometrymap
    # 组合x, y, z坐标形成点云
    pointcloud = torch.cat((x*valid_ids, y*valid_ids, z*valid_ids), dim=1)
    if not keep_dim:
        pointcloud = pointcloud.view(B, 6, -1).permute(0, 2, 1)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pointcloud[0].cpu().numpy())
    # o3d.visualization.draw_geometries([pcd])

    return pointcloud


def get_rot_matrix(angle_x, angle_y, angle_z):
    angle_x = angle_x / 180.0 * np.pi
    angle_y = angle_y / 180.0 * np.pi
    angle_z = angle_z / 180.0 * np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])

    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R


def fps(pc, n_point):
    pc_pt = torch.from_numpy(pc).contiguous().unsqueeze(0).float().cuda()
    pc_sampled = fps_subsample_v2(pc_pt, n_points=n_point)
    pc_sampled_arr = pc_sampled.squeeze(0).detach().cpu().numpy()
    return pc_sampled_arr.astype(np.float32)


def points_normalize(pc, factor, offset=0.0):
    bmin, bmax = np.min(pc, axis=0), np.max(pc, axis=0)
    center, scale = (bmin+bmax)/2., np.max(bmax-bmin)
    pc -= center
    pc *= factor / scale
    pc += offset

    return pc, center, scale


def crop_and_resize(data_tensor, mask_tensor):
    B, C, H, W = data_tensor.shape
    cropped_resized_data = torch.zeros_like(data_tensor)

    for b in range(B):
        mask = mask_tensor[b, 0]
        non_zero_indices = torch.nonzero(mask, as_tuple=True)

        if non_zero_indices[0].size(0) == 0:  # 检查是否有非零元素
            continue  # 如果当前掩码全为零，则跳过

        min_h, max_h = non_zero_indices[0].min(), non_zero_indices[0].max()
        min_w, max_w = non_zero_indices[1].min(), non_zero_indices[1].max()

        # 计算裁剪区域的高和宽
        crop_height = max_h - min_h + 1
        crop_width = max_w - min_w + 1

        # 确定保持长宽比的新尺寸
        aspect_ratio = crop_height.float() / crop_width.float()
        if crop_height > crop_width:
            new_height = H
            new_width = torch.round(H / aspect_ratio).int().item()
        else:
            new_width = W
            new_height = torch.round(W * aspect_ratio).int().item()

        # 裁剪并调整尺寸
        cropped_data = data_tensor[b, :, min_h:max_h + 1, min_w:max_w + 1]
        resized_data = F.interpolate(cropped_data.unsqueeze(0), size=(new_height, new_width), mode='bilinear',
                                     align_corners=True)

        # 由于裁剪后尺寸可能与原始尺寸不匹配，需要进行中心对齐
        padded_data = torch.zeros((1, C, H, W), device=data_tensor.device)
        start_x = (W - new_width) // 2
        start_y = (H - new_height) // 2
        padded_data[:, :, start_y:start_y + new_height, start_x:start_x + new_width] = resized_data

        cropped_resized_data[b] = padded_data.squeeze(0)

    return cropped_resized_data


def translate_to_match_centroid(A_pre, B_pre, mask_pre, mask_gt):
    # 计算pre掩码的质心
    indices_pre = torch.nonzero(mask_pre[0], as_tuple=True)
    centroid_y_pre, centroid_x_pre = torch.mean(indices_pre[0].float()), torch.mean(indices_pre[1].float())
    # 计算gt掩码的质心
    indices_gt = torch.nonzero(mask_gt[0], as_tuple=True)
    centroid_y_gt, centroid_x_gt = torch.mean(indices_gt[0].float()), torch.mean(indices_gt[1].float())
    # 计算质心之间的位移差
    dy, dx = int(centroid_y_gt.item() - centroid_y_pre.item()), int(centroid_x_gt.item() - centroid_x_pre.item())
    # 创建平移后的图像张量
    translated_A_pre = torch.zeros_like(A_pre)
    translated_B_pre = torch.zeros_like(B_pre)
    # 计算平移后的索引范围
    _, _, H, W = A_pre.shape
    y1, y2 = max(0, dy), min(H, H + dy)
    x1, x2 = max(0, dx), min(W, W + dx)
    # 将原始图像的对应部分复制到新位置
    translated_A_pre[:, :, y1:y2, x1:x2] = A_pre[:, :, max(0, -dy):min(H, H - dy), max(0, -dx):min(W, W - dx)]
    translated_B_pre[:, :, y1:y2, x1:x2] = B_pre[:, :, max(0, -dy):min(H, H - dy), max(0, -dx):min(W, W - dx)]
    return translated_A_pre, translated_B_pre


def evaluate_matrics_old(depth_pre, normal_pre, depth_gt, normal_gt, translate=False):
    normal_norm_pre = torch.norm(normal_pre, p=2, dim=1, keepdim=True)
    normal_norm_gt = torch.norm(normal_gt, p=2, dim=1, keepdim=True)
    mask_pre = normal_norm_pre >= 0.5
    mask_gt = normal_norm_gt >= 0.5
    mask_cal = mask_pre | mask_gt

    if translate:
        depth_pre, normal_pre = translate_to_match_centroid(depth_pre, normal_pre, mask_pre, mask_gt)
        # depth_pre = crop_and_resize(depth_pre, mask_pre)
        # normal_pre = crop_and_resize(normal_pre, mask_pre)
        # depth_gt = crop_and_resize(depth_gt, mask_gt)
        # normal_gt = crop_and_resize(normal_gt, mask_gt)

    depth_fig_pre, normal_fig_pre = to_visual_image(depth_pre, normal_pre)
    depth_fig_gt, normal_fig_gt = to_visual_image(depth_gt, normal_gt)

    RMSE = RMSEMetric()

    depth_PSNR = PSNRMetric(max_val=255)
    normal_PSNR = PSNRMetric(max_val=255)

    depth_SSIM = SSIMMetric(spatial_dims=2)
    normal_SSIM = SSIMMetric(spatial_dims=2)

    # 添加小偏置来防止除0
    normal_norm_pre = torch.clamp(normal_norm_pre, min=1e-6)
    normal_norm_gt = torch.clamp(normal_norm_gt, min=1e-6)
    cos_angle = (normal_pre * normal_gt).sum(dim=1, keepdim=True) / (normal_norm_pre * normal_norm_gt)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    if not translate:
        # 计算法向图的平均夹角 将gt和预测法向图的mask掩码并集来计算
        angles = torch.acos(cos_angle) * mask_cal
        mean_angle_deg_masked = torch.mean(torch.rad2deg(angles)[mask_cal]).repeat(angles.shape[0], 1)
        mean_angle_deg_all = torch.mean(torch.rad2deg(angles), dim=(2, 3))
    else:
        angles = torch.acos(cos_angle)
        mean_angle_deg_all = torch.mean(torch.rad2deg(angles), dim=(2, 3))
        mean_angle_deg_masked = mean_angle_deg_all

    # 将深度图转换成点云并且计算倒角距离
    # 计算的成本过大 建议另存出点云然后单独计算
    # pc_pre = depth_to_pointcloud(depth_pre, fx=fx, fy=fy, cx=cx, cy=cy).permute(0, 2, 1)
    # pc_gt = depth_to_pointcloud(depth_gt, fx=fx, fy=fy, cx=cx, cy=cy).permute(0, 2, 1)
    # pc_chamferdist = ChamferDistance(pc_pre, pc_gt)

    thre_edge = [-1, 1]
    thre_curv = [22, 24]
    edge_pre = laplacianConv(depth_pre)
    edge_gt = laplacianConv(depth_gt)
    curv_pre = calCurvature(normal_pre)
    curv_gt = calCurvature(normal_gt)
    edge_pre = (edge_pre.clip(thre_edge[0], thre_edge[1]) - thre_edge[0])
    edge_gt = (edge_gt.clip(thre_edge[0], thre_edge[1]) - thre_edge[0])
    curv_pre = (curv_pre.clip(thre_curv[0], thre_curv[1]) - thre_curv[0])
    curv_gt = (curv_gt.clip(thre_curv[0], thre_curv[1]) - thre_curv[0])

    return [RMSE(depth_pre, depth_gt), RMSE(normal_pre, normal_gt),
            depth_PSNR(depth_fig_pre, depth_fig_gt), normal_PSNR(normal_fig_pre, normal_fig_gt),
            depth_SSIM(depth_fig_pre, depth_fig_gt), normal_SSIM(normal_fig_pre, normal_fig_gt),
            mean_angle_deg_masked, mean_angle_deg_all,
            RMSE(edge_pre, edge_gt), RMSE(curv_pre, curv_gt)]


def evaluate_matrics(depth_pre, normal_pre, depth_gt, normal_gt, translate=False):
    normal_norm_pre = torch.norm(normal_pre, p=2, dim=1, keepdim=True)
    normal_norm_gt = torch.norm(normal_gt, p=2, dim=1, keepdim=True)
    mask_pre = normal_norm_pre >= 0.5
    mask_gt = normal_norm_gt >= 0.5
    mask_cal = mask_pre | mask_gt

    if translate:
        depth_pre, normal_pre = translate_to_match_centroid(depth_pre, normal_pre, mask_pre, mask_gt)
        # depth_pre = crop_and_resize(depth_pre, mask_pre)
        # normal_pre = crop_and_resize(normal_pre, mask_pre)
        # depth_gt = crop_and_resize(depth_gt, mask_gt)
        # normal_gt = crop_and_resize(normal_gt, mask_gt)

    depth_fig_pre, normal_fig_pre = to_visual_image(depth_pre, normal_pre)
    depth_fig_gt, normal_fig_gt = to_visual_image(depth_gt, normal_gt)

    RMSE = RMSEMetric()

    depth_PSNR = PSNRMetric(max_val=255)
    normal_PSNR = PSNRMetric(max_val=255)

    depth_SSIM = SSIMMetric(spatial_dims=2)
    normal_SSIM = SSIMMetric(spatial_dims=2)

    # 添加小偏置来防止除0
    normal_norm_pre = torch.clamp(normal_norm_pre, min=1e-6)
    normal_norm_gt = torch.clamp(normal_norm_gt, min=1e-6)
    cos_angle = (normal_pre * normal_gt).sum(dim=1, keepdim=True) / (normal_norm_pre * normal_norm_gt)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    if not translate:
        # 计算法向图的平均夹角 将gt和预测法向图的mask掩码并集来计算
        angles = torch.acos(cos_angle) * mask_cal
        mean_angle_deg_masked = torch.mean(torch.rad2deg(angles)[mask_cal]).repeat(angles.shape[0], 1)
        mean_angle_deg_all = torch.mean(torch.rad2deg(angles), dim=(2, 3))
    else:
        angles = torch.acos(cos_angle)
        mean_angle_deg_all = torch.mean(torch.rad2deg(angles), dim=(2, 3))
        mean_angle_deg_masked = mean_angle_deg_all

    # 将深度图转换成点云并且计算倒角距离
    # 计算的成本过大 建议另存出点云然后单独计算
    # pc_pre = depth_to_pointcloud(depth_pre, fx=fx, fy=fy, cx=cx, cy=cy).permute(0, 2, 1)
    # pc_gt = depth_to_pointcloud(depth_gt, fx=fx, fy=fy, cx=cx, cy=cy).permute(0, 2, 1)
    # pc_chamferdist = ChamferDistance(pc_pre, pc_gt)

    thre_edge = [-1, 1]
    thre_curv = [22, 24]
    edge_pre = laplacianConv(depth_pre)
    edge_gt = laplacianConv(depth_gt)
    curv_pre = calCurvature(normal_pre)
    curv_gt = calCurvature(normal_gt)
    edge_pre = (edge_pre.clip(thre_edge[0], thre_edge[1]) - thre_edge[0])
    edge_gt = (edge_gt.clip(thre_edge[0], thre_edge[1]) - thre_edge[0])
    curv_pre = (curv_pre.clip(thre_curv[0], thre_curv[1]) - thre_curv[0])
    curv_gt = (curv_gt.clip(thre_curv[0], thre_curv[1]) - thre_curv[0])

    return [RMSE(depth_pre, depth_gt), RMSE(normal_pre, normal_gt),
            depth_PSNR(depth_fig_pre, depth_fig_gt), normal_PSNR(normal_fig_pre, normal_fig_gt),
            depth_SSIM(depth_fig_pre, depth_fig_gt), normal_SSIM(normal_fig_pre, normal_fig_gt),
            mean_angle_deg_masked, mean_angle_deg_all,
            RMSE(edge_pre, edge_gt), RMSE(curv_pre, curv_gt)]


def calculate_fid(map_pred, map_gt):
    """
    计算两组特征映射的FID分数。
    :param map_pred: 预测的特征映射，形状为Bx3xHxW
    :param map_gt: 真实的特征映射，形状为Bx3xHxW
    :return: FID分数
    """
    device = map_pred.device
    B, N, H, W = map_pred.shape
    if N == 1:
        map_pred = map_pred.repeat(1, 3, 1, 1)
        map_gt = map_gt.repeat(1, 3, 1, 1)
    # 加载Inception模型
    inception_model = inception_v3(pretrained=True).to(device)

    # 获取特征并转换为NumPy数组
    inception_model.eval()
    with torch.no_grad():
        pred_features = inception_model(map_pred).view(B, -1).cpu().numpy()
        gt_features = inception_model(map_gt).view(B, -1).cpu().numpy()

    # 计算均值和协方差
    mu_pred, cov_pred = np.mean(pred_features, axis=0), np.cov(pred_features, rowvar=False)
    mu_gt, cov_gt = np.mean(gt_features, axis=0), np.cov(gt_features, rowvar=False)

    # 计算FID
    ssdiff = np.sum((mu_pred - mu_gt) ** 2)
    covmean = sqrtm(cov_pred.dot(cov_gt))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(cov_pred + cov_gt - 2 * covmean)

    return fid


class evaluate_metrics_list:
    def __init__(self, default_device):
        from torchvision.models import inception_v3
        import lpips
        from monai.metrics import RMSEMetric, PSNRMetric, SSIMMetric, FIDMetric

        self.rmse = RMSEMetric()
        self.psnr = PSNRMetric(max_val=255)
        self.ssim = SSIMMetric(spatial_dims=2)

        self.inception = inception_v3(pretrained=True).to(default_device).eval()
        self.inception.requires_grad_(False)
        self.fid = FIDMetric()
        self.lpips = lpips.LPIPS(net='alex').to(default_device).eval()
        self.lpips.requires_grad_(False)

        # 用编号索引元组 [0]名称 [1]以batch为单位的结果列表 [2]均值
        self.metrics_list = {0: ("RMSE_depth", [], [0.0]),
                             1: ("RMSE_normal", [], [0.0]),
                             2: ("Angle_mean_all", [], [0.0]),
                             3: ("Angle_mean_union", [], [0.0]),
                             4: ("Angle_mean_inter", [], [0.0]),
                             5: ("PSNR_depth", [], [0.0]),
                             6: ("PSNR_normal", [], [0.0]),
                             7: ("SSIM_depth", [], [0.0]),
                             8: ("SSIM_normal", [], [0.0]),
                             9: ("RMSE_edge", [], [0.0]),
                             10: ("RMSE_curv", [], [0.0]),
                             11: ("FID_depth", [], [0.0]),
                             12: ("FID_normal", [], [0.0]),
                             13: ("LPIPS_depth", [], [0.0]),
                             14: ("LPIPS_normal", [], [0.0]),
                             }

    def get_evaluate_metrics(self, depth_pre, normal_pre, depth_gt, normal_gt, normalized=True):
        B = depth_pre.shape[0]
        # 计算mask
        normal_norm_pre = torch.norm(normal_pre, p=2, dim=1, keepdim=True)
        normal_norm_gt = torch.norm(normal_gt, p=2, dim=1, keepdim=True)
        mask_pre = normal_norm_pre >= 0.5
        mask_gt = normal_norm_gt >= 0.5
        # mask交/并可选项
        mask_u = mask_pre | mask_gt
        mask_i = mask_pre & mask_gt

        if normalized:
            normalized_normal_pre = (normal_pre / normal_norm_pre) * mask_pre
            normalized_depth_pre = depth_pre * mask_pre
        else:
            normalized_normal_pre = normal_pre
            normalized_depth_pre = depth_pre

        ### 计算基于重建的指标 ###
        RMSE_depth = self.rmse(normalized_depth_pre, depth_gt)
        RMSE_normal = self.rmse(normalized_normal_pre, normal_gt)

        ### 计算法向角度误差 ###
        # 用[0,0,1]替换法向的空值
        z = torch.zeros_like(normal_gt)
        z[:, 2, :, :] = 1.0
        normal_pre_angle = normalized_normal_pre
        normal_pre_angle = torch.where(~mask_pre, z, normal_pre_angle)
        normal_gt_angle = torch.where(~mask_gt, z, normal_gt)
        # 计算全图的法向夹角
        cos_angle = (normal_pre_angle * normal_gt_angle).sum(dim=1, keepdim=True)
        angles = torch.rad2deg(torch.acos(cos_angle))
        Angle_mean_all = torch.mean(angles, dim=(2, 3))
        # 计算mask交并法向夹角
        Angle_mean_union = (torch.sum(angles[torch.where(mask_u)]) / mask_u.sum()).repeat(B, 1)
        Angle_mean_inter = (torch.sum(angles[torch.where(mask_i)]) / mask_i.sum()).repeat(B, 1)

        ### 转换成为图片 ###
        depth_fig_pre = to_visual_image_single(normalized_depth_pre, mask_pre, threshold=(-2, 2))
        normal_fig_pre = to_visual_image_single(normalized_normal_pre, mask_pre, threshold=(-1, 1))
        depth_fig_gt = to_visual_image_single(depth_gt, mask_gt, threshold=(-2, 2))
        normal_fig_gt = to_visual_image_single(normal_gt, mask_gt, threshold=(-1, 1))

        ### 计算基于图片的matrics ###
        PSNR_depth = self.psnr(depth_fig_pre, depth_fig_gt)
        PSNR_normal = self.psnr(normal_fig_pre, normal_fig_gt)
        SSIM_depth = self.ssim(depth_fig_pre, depth_fig_gt)
        SSIM_normal = self.ssim(normal_fig_pre, normal_fig_gt)

        ### 利用laplacianConv算子计算edge ###
        thre_edge = [-1, 1]
        edge_pre = laplacianConv(normalized_depth_pre)
        edge_gt = laplacianConv(depth_gt)
        edge_pre = (edge_pre.clip(thre_edge[0], thre_edge[1]) - thre_edge[0])
        edge_gt = (edge_gt.clip(thre_edge[0], thre_edge[1]) - thre_edge[0])
        RMSE_edge = self.rmse(edge_pre, edge_gt)

        ### 利用calCurvature算子计算curv ###
        thre_curv = [22, 24]
        curv_pre = calCurvature(normalized_normal_pre)
        curv_gt = calCurvature(normal_gt)
        curv_pre = (curv_pre.clip(thre_curv[0], thre_curv[1]) - thre_curv[0])
        curv_gt = (curv_gt.clip(thre_curv[0], thre_curv[1]) - thre_curv[0])
        RMSE_curv = self.rmse(curv_pre, curv_gt)

        ### 计算FID_depth ###
        feat_depth_pre = self.inception(depth_fig_pre.repeat(1, 3, 1, 1)).view(B, -1)
        feat_depth_gt = self.inception(depth_fig_gt.repeat(1, 3, 1, 1)).view(B, -1)
        FID_depth = self.fid(feat_depth_pre, feat_depth_gt).view(1, 1)

        ### 计算FID_normal ###
        feat_norm_pre = self.inception(normal_fig_pre).view(B, -1)
        feat_norm_gt = self.inception(normal_fig_gt).view(B, -1)
        FID_normal = self.fid(feat_norm_pre, feat_norm_gt).view(1, 1)

        ### 计算LPIPS ###
        LPIPS_depth = self.lpips(depth_fig_pre, depth_fig_gt).view(B, 1)
        LPIPS_normal = self.lpips(normal_fig_pre, normal_fig_gt).view(B, 1)

        return [RMSE_depth,
                RMSE_normal,
                Angle_mean_all,
                Angle_mean_union,
                Angle_mean_inter,
                PSNR_depth,
                PSNR_normal,
                SSIM_depth,
                SSIM_normal,
                RMSE_edge,
                RMSE_curv,
                FID_depth,
                FID_normal,
                LPIPS_depth,
                LPIPS_normal]

    def update_metrics_list(self, new_metrics_list):
        for i in range(len(self.metrics_list)):
            # 更新列表
            self.metrics_list[i][1].append(new_metrics_list[i])
            # 更新均值
            self.metrics_list[i][2][0] = torch.cat(self.metrics_list[i][1], dim=0).mean()

    def print_metrics_list(self):
        for i in range(len(self.metrics_list)):
            # 更新均值
            self.metrics_list[i][2][0] = torch.cat(self.metrics_list[i][1], dim=0).mean()
            # print('%s: %f' % (self.metrics_list[i][0], self.metrics_list[i][2][0]))
            print('%f' % (self.metrics_list[i][2][0]))


def save_model_output(result_list, name, depth_map, normal_map, save_path, normalized=True, depth_threshold=(-2, 2)):
    B = len(name)
    normal_norm = torch.norm(normal_map, p=2, dim=1, keepdim=True)
    mask = normal_norm >= 0.5
    for b in range(B):
        save_path.joinpath(name[b]).mkdir(exist_ok=True)

    if normalized:
        normal_map = (normal_map / normal_norm) * mask
        depth_map = depth_map * mask

    if result_list["Source"]:
        depth_source = depth_map.cpu().numpy()
        normal_source = normal_map.cpu().numpy()
        mask_source = mask.cpu().numpy()
        for b in range(B):
            np.save(str(save_path.joinpath(name[b] + '/depth_src.npy')), depth_source[b])
            np.save(str(save_path.joinpath(name[b] + '/normal_src.npy')), normal_source[b])
            np.save(str(save_path.joinpath(name[b] + '/mask_src.npy')), mask_source[b])

    if result_list["Figure"]:
        depth_fig = to_visual_image_single(depth_map, mask, threshold=depth_threshold).permute(0, 2, 3, 1).cpu().numpy()
        normal_fig = to_visual_image_single(normal_map, mask, threshold=(-1, 1)).permute(0, 2, 3, 1).cpu().numpy()
        mask_fig = mask.permute(0, 2, 3, 1).cpu().numpy() * 255
        for b in range(B):
            cv2.imwrite(str(save_path.joinpath(name[b] + '/depth_fig.png')), depth_fig[b])
            cv2.imwrite(str(save_path.joinpath(name[b] + '/normal_fig.png')), normal_fig[b])
            cv2.imwrite(str(save_path.joinpath(name[b] + '/mask_fig.png')), mask_fig[b])

    if result_list["Pointcloud"]:
        pc_all = geometrymap_to_pointcloud(torch.cat([depth_map, normal_map], dim=1), fx=16, fy=16, cx=256, cy=256)
        for b in range(B):
            pc = pc_all[b][mask[b].view(-1)].cpu().numpy()
            np.savetxt(str(save_path.joinpath(name[b] + '/pointcloud.txt')), pc)
            if result_list["Mesh"]:
                vert, triangles = depth_to_mesh((512, 512), mask[b, 0].cpu().numpy())
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(pc[:, :3])
                mesh.triangles = o3d.utility.Vector3iVector(triangles)

                mesh = mesh.simplify_quadric_decimation(100000)
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.remove_duplicated_vertices()
                mesh.remove_non_manifold_edges()
                o3d.io.write_triangle_mesh(str(save_path.joinpath(name[b] + '/mesh.ply')), mesh)

    if result_list["Edge"]:
        thre_edge = [-1, 1]
        edge = laplacianConv(depth_map)
        edge = (edge.clip(thre_edge[0], thre_edge[1]) - thre_edge[0]) / (thre_edge[1] - thre_edge[0])
        edge = edge.permute(0, 2, 3, 1).cpu().numpy() * 255
        for b in range(B):
            cv2.imwrite(str(save_path.joinpath(name[b] + '/edge_fig.png')), edge[b])

    if result_list["Curv"]:
        thre_curv = [22, 24]
        curv = calCurvature(normal_map)
        curv = (curv.clip(thre_curv[0], thre_curv[1]) - thre_curv[0]) / (thre_curv[1] - thre_curv[0])
        curv = curv.permute(0, 2, 3, 1).cpu().numpy() * 255
        for b in range(B):
            cv2.imwrite(str(save_path.joinpath(name[b] + '/curv_fig.png')), curv[b])

    if result_list["Sobel"]:
        thre_sobel = [-1, 1]
        dz_dx, dz_dy, normal_cal = sobelConv(depth_map, mask)
        dz_dx = (dz_dx.clip(thre_sobel[0], thre_sobel[1]) - thre_sobel[0]) / (thre_sobel[1] - thre_sobel[0])
        dz_dx = dz_dx.permute(0, 2, 3, 1).cpu().numpy() * 255
        dz_dy = (dz_dy.clip(thre_sobel[0], thre_sobel[1]) - thre_sobel[0]) / (thre_sobel[1] - thre_sobel[0])
        dz_dy = dz_dy.permute(0, 2, 3, 1).cpu().numpy() * 255
        normal_cal = to_visual_image_single(normal_cal, mask, threshold=(-1, 1)).permute(0, 2, 3, 1).cpu().numpy()
        for b in range(B):
            cv2.imwrite(str(save_path.joinpath(name[b] + '/sobel_x_fig.png')), dz_dx[b])
            cv2.imwrite(str(save_path.joinpath(name[b] + '/sobel_y_fig.png')), dz_dy[b])
            cv2.imwrite(str(save_path.joinpath(name[b] + '/cal_normal_fig.png')), normal_cal[b])







def depth_to_voxel(depth_map, mask, scale=16, size=256):
    """
    将深度图转换为体素表示。
    :param depth_map: 输入的深度图，形状为Bx1xHxW
    :param mask: 输入的掩码，形状为Bx1xHxW
    :param scale: 深度缩放因子
    :param size: 体素空间的尺寸
    :return: 体素表示，形状为Bx1xsizexsizexsize
    """
    B, _, H, W = depth_map.shape
    depth_scale_factor = scale * (size / H)

    # 缩放深度图
    depth_map_scaled = depth_map * depth_scale_factor
    # 调整深度图和掩码大小以匹配体素空间的尺寸
    depth_map_resized = F.interpolate(depth_map_scaled, size=(size, size), mode='bilinear', align_corners=True)
    mask_resized = F.interpolate(mask.float(), size=(size, size), mode='nearest').byte().squeeze(1)
    # 转换深度值为体素索引
    depth_indices = (depth_map_resized + size / 2).clamp(0, size - 1).long()
    # 创建体素表示
    voxel = torch.zeros((B, 1, size, size, size), device=depth_map.device)

    # 使用批量操作设置体素值
    for b in range(B):
        b_mask = mask_resized[b]
        # x_indices, y_indices = torch.where(b_mask)
        # z_indices = depth_indices[b, 0, x_indices, y_indices]
        z_values = depth_indices[b, 0]
        z_matrix = torch.arange(size).view(1, 1, -1).expand(size, size, size).to(depth_map.device)
        mask_3d = (z_matrix < z_values.unsqueeze(-1)) & b_mask.unsqueeze(-1)
        voxel[b, 0][mask_3d.bool()] = 1
        # voxel[b, 0, x_indices, y_indices, z_indices] = 1.0
        # voxel[b, 0, x_indices, y_indices, z_indices + 1] = 1.0
        # voxel[b, 0, x_indices, y_indices, z_indices - 1] = 1.0
        # voxel[b, 0, x_indices, y_indices, z_indices] = 1.0
        # for i, x in enumerate(x_indices):
        #     voxel[b, 0, x_indices[i], y_indices[i], :z_indices[i]] = 1.0
    return voxel


def voxel_to_depth(voxel, scale=16, size=256, original_size=(512, 512), threshold=0.5, return_mask=False):
    """
    将体素表示转换为深度图，对于重叠的体素，取z坐标最大的值。
    :param voxel: 输入的体素，形状为Bx1xsizexsizexsize
    :param scale: 深度缩放因子
    :param size: 体素空间的尺寸
    :param original_size: 原始深度图的尺寸（H, W）
    :return: 深度图，形状为Bx1xHxW
    """
    B, _, _, _, _ = voxel.shape
    H, W = original_size

    # 初始化深度图
    depth_map = torch.full((B, 1, size, size), 0, dtype=torch.float32, device=voxel.device)
    # 初始化mask
    depth_mask = torch.full((B, 1, size, size), 0, dtype=torch.float32, device=voxel.device)
    # 计算深度缩放因子
    depth_scale_factor = scale * (size / H)
    # 提取体素中的深度信息
    for b in range(B):
        _, x_indices, y_indices, z_indices = torch.where(voxel[b] > threshold)
        # 将体素索引转换为深度值
        depth_values = (z_indices.float() - size / 2) / depth_scale_factor
        # 对每个(x, y)坐标组选择z坐标最大的值
        xy_indices = torch.stack((x_indices, y_indices), dim=1)
        unique_xy, inverse_indices = torch.unique(xy_indices, return_inverse=True, dim=0)
        max_z_vals = torch.zeros((unique_xy.shape[0],), device=voxel.device)
        for i in range(unique_xy.shape[0]):
            mask = (inverse_indices == i)
            max_z_vals[i] = torch.max(depth_values[mask])
        depth_map[b, 0, unique_xy[:, 0], unique_xy[:, 1]] = max_z_vals
        depth_mask[b, 0, unique_xy[:, 0], unique_xy[:, 1]] = 1
    # 调整深度图大小以匹配原始尺寸
    depth_map_resized = F.interpolate(depth_map, size=original_size, mode='bilinear', align_corners=True)
    depth_mask_resized = F.interpolate(depth_mask, size=original_size, mode='nearest')

    if return_mask:
        return depth_map_resized, depth_mask_resized
    else:
        return depth_map_resized


def laplacianConv(depth_pre, channels=1):
    device = depth_pre.device
    kernel = torch.FloatTensor([[-1.0, -1.0, -1.0, -1.0, -1.0],
                                [-1.0, -1.0, -1.0, -1.0, -1.0],
                                [-1.0, -1.0, 24.0, -1.0, -1.0],
                                [-1.0, -1.0, -1.0, -1.0, -1.0],
                                [-1.0, -1.0, -1.0, -1.0, -1.0]]) * 0.5
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    kernel = np.repeat(kernel, channels, axis=0)
    weight = torch.nn.Parameter(data=kernel, requires_grad=False).to(device)
    return torch.nn.functional.conv2d(depth_pre, weight, padding=2, groups=channels)


def calCurvature(normal_pre, image_size=[512, 512]):
    device = normal_pre.device
    W, H = image_size
    W = W + 4
    H = H + 4
    # padding的参数为左右上下方向填充的行数
    ZeroPad = torch.nn.ZeroPad2d(padding=(2, 2, 2, 2)).to(device)
    # 设置24邻域进行计算的边界 每个list中四个元素分别是x轴左、右界 y轴下、上界
    index_bound = [[0, W - 4, 0, H - 4], [1, W - 3, 0, H - 4], [2, W - 2, 0, H - 4], [3, W - 1, 0, H - 4],
                   [4, W - 0, 0, H - 4],
                   [0, W - 4, 1, H - 3], [1, W - 3, 1, H - 3], [2, W - 2, 1, H - 3], [3, W - 1, 1, H - 3],
                   [4, W - 0, 1, H - 3],
                   [0, W - 4, 2, H - 2], [1, W - 3, 2, H - 2], [3, W - 1, 2, H - 2], [4, W - 0, 2, H - 2],
                   [0, W - 4, 3, H - 1], [1, W - 3, 3, H - 1], [2, W - 2, 3, H - 1], [3, W - 1, 3, H - 1],
                   [4, W - 0, 3, H - 1],
                   [0, W - 4, 4, H - 0], [1, W - 3, 4, H - 0], [2, W - 2, 4, H - 0], [3, W - 1, 4, H - 0],
                   [4, W - 0, 4, H - 0]]
    pad_x = ZeroPad(normal_pre)
    cos_theta = torch.zeros_like(normal_pre[:, :1, :, :])
    for bound in index_bound:
        # 计算5*5邻域中某一个像素的点乘
        cos_theta += torch.sum(normal_pre * pad_x[:, :, bound[0]:bound[1], bound[2]:bound[3]], dim=1, keepdim=True)
    return cos_theta


def sobelConv(depth_pre, mask, channels=1):
    device = depth_pre.device
    sobel_x = torch.FloatTensor([[[[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]]]])
    sobel_x = torch.nn.Parameter(data=sobel_x, requires_grad=False).to(device)
    sobel_y = torch.FloatTensor([[[[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]]]])
    sobel_y = torch.nn.Parameter(data=sobel_y, requires_grad=False).to(device)

    dz_dx = torch.nn.functional.conv2d(depth_pre, sobel_x, padding=1)
    dz_dy = torch.nn.functional.conv2d(depth_pre, sobel_y, padding=1)

    normal_cal = torch.cat((-dz_dx, -dz_dy, torch.ones_like(dz_dx) * 0.5), dim=1)
    norm = torch.norm(normal_cal, p=2, dim=1, keepdim=True)
    normal_cal = normal_cal / norm
    normal_cal = normal_cal * mask
    return dz_dx, dz_dy, normal_cal


def depth_to_mesh(image_shape, mask):
    rows, cols = image_shape
    Y, X = np.mgrid[0:rows, 0:cols]

    # 使用掩码过滤点
    valid_points = mask.ravel()
    vertices = np.stack([X.ravel()[valid_points], Y.ravel()[valid_points]], axis=1)

    # 构建索引映射，将掩码中的索引映射到新的顶点数组中
    index_map = np.full((rows, cols), -1, dtype=int)
    index_map.ravel()[valid_points] = np.arange(np.sum(valid_points))

    # 生成三角形元素
    triangles = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            if mask[r, c] and mask[r, c + 1] and mask[r + 1, c]:
                # 第一个三角形 ABC
                triangles.append([index_map[r, c], index_map[r, c + 1], index_map[r + 1, c]])
            if mask[r, c + 1] and mask[r + 1, c] and mask[r + 1, c + 1]:
                # 第二个三角形 BCD
                triangles.append([index_map[r, c + 1], index_map[r + 1, c + 1], index_map[r + 1, c]])
            if not mask[r, c + 1] and mask[r, c] and mask[r + 1, c] and mask[r + 1, c + 1]:
                # 如果不满足前两个三角形 判断能否反向构建
                triangles.append([index_map[r, c], index_map[r + 1, c + 1], index_map[r + 1, c]])
            if not mask[r + 1, c] and mask[r, c] and mask[r, c + 1] and mask[r + 1, c + 1]:
                # 如果不满足前两个三角形 判断能否反向构建
                triangles.append([index_map[r, c], index_map[r, c + 1], index_map[r + 1, c + 1]])

    return vertices, np.array(triangles)


def depth_to_normal(depth_map, mask=None):
    # 使用Sobel滤波器计算深度图的梯度
    sobel_x = torch.tensor([[[[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]]]], dtype=torch.float32, requires_grad=False, device=depth_map.device)
    sobel_y = torch.tensor([[[[-1, -2, -1],
                              [0, 0, 0],
                              [1, 2, 1]]]], dtype=torch.float32, requires_grad=False, device=depth_map.device)

    dz_dx = F.conv2d(depth_map, sobel_x, padding=1)
    dz_dy = F.conv2d(depth_map, sobel_y, padding=1)

    # 计算法向量
    normal_map = torch.cat((-dz_dx, -dz_dy, torch.ones_like(dz_dx) * 0.5), dim=1)
    # normal_map = torch.cat((-dz_dx, -dz_dy, torch.ones_like(dz_dx) * (2 / fx)), dim=1)

    # 归一化法向量
    # norm = torch.sqrt(torch.sum(normal_map ** 2, dim=1, keepdim=True))
    norm = torch.norm(normal_map, p=2, dim=1, keepdim=True)
    normal_map = normal_map / norm

    if not mask == None:
        normal_map = normal_map * mask

    return normal_map

# depth = cv2.imread('../temp/pair_34_depth_gt.png', cv2.IMREAD_GRAYSCALE)
# normal = cv2.imread('../temp/pair_34_normal_gt.png', cv2.IMREAD_COLOR)
# depth = depth.astype(np.float32)
# normal_cal = depth_to_normal(depth)
# noraml_cal = (normal_cal + 1) / 2
# cv2.imshow('Depth Map', depth / depth.max())  # 归一化显示深度图
# cv2.imshow('Normal Map', normal)
# cv2.imshow('Normal Map Cal', noraml_cal)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
