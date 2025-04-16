import json
import logging
import numpy as np
import random
import torch.utils.data.dataset
import open3d as o3d
import utils.data_transforms
import torchvision.transforms as trans
from enum import Enum, unique
from tqdm import tqdm
from utils.io import IO
import os
import re
from PIL import Image

label_mapping = {
    3: '03001627',
    6: '04379243',
    5: '04256520',
    1: '02933112',
    4: '03636649',
    2: '02958343',
    0: '02691156',
    7: '04530566'
}


@unique
class DatasetSubset(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


def collate_fn(batch):
    taxonomy_ids = []
    model_ids = []
    data = {}

    for sample in batch:
        taxonomy_ids.append(sample[0])
        model_ids.append(sample[1])
        _data = sample[2]
        for k, v in _data.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k, v in data.items():
        data[k] = torch.stack(v, 0)

    return taxonomy_ids, model_ids, data


def collate_fn_55(batch):
    taxonomy_ids = []
    model_ids = []
    data = {}

    for sample in batch:
        taxonomy_ids.append(sample[0])
        model_ids.append(sample[1])
        _data = sample[2]
        for k, v in _data.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k, v in data.items():
        data[k] = torch.stack(v, 0)

    return taxonomy_ids, model_ids, data


def collate_fn_depth(batch):
    taxonomy_ids = []
    model_ids = []
    data = {}

    for sample in batch:
        taxonomy_ids.append(sample[0])
        model_ids.append(sample[1])
        _data = sample[2]
        for k, v in _data.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k, v in data.items():
        data[k] = torch.stack(v, 0)

    return taxonomy_ids, model_ids, data


code_mapping = {
    'plane': '02691156',
    'cabinet': '02933112',
    'car': '02958343',
    'chair': '03001627',
    'lamp': '03636649',
    'couch': '04256520',
    'table': '04379243',
    'watercraft': '04530566',
}


def read_ply(file_path):
    pc = o3d.io.read_point_cloud(file_path)
    ptcloud = np.array(pc.points)
    return ptcloud


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, options, file_list, transforms=None):
        self.options = options
        self.file_list = file_list
        self.transforms = transforms
        self.cache = dict()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = -1
        if 'n_renderings' in self.options:
            rand_idx = random.randint(0, self.options['n_renderings'] - 1) if self.options['shuffle'] else 0

        for ri in self.options['required_items']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            data[ri] = IO.get(file_path).astype(np.float32)

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], data


class ShapeNetDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.SHAPENET.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        n_renderings = self.cfg.DATASETS.SHAPENET.N_RENDERINGS if subset == DatasetSubset.TRAIN else 1
        file_list = self._get_file_list(self.cfg, self._get_subset(subset), n_renderings)
        transforms = self._get_transforms(self.cfg, subset)
        return Dataset({
            'n_renderings': n_renderings,
            'required_items': ['partial_cloud', 'gtcloud'],
            'shuffle': subset == DatasetSubset.TRAIN
        }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):

                if subset == 'test':

                    gt_path = cfg.DATASETS.SHAPENET.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s)
                    file_list.append({'taxonomy_id': dc['taxonomy_id'],
                                      'model_id': s,
                                      'partial_cloud_path': gt_path.replace('complete', 'partial')[
                                                            :-4] + '/00' + gt_path.replace('complete', 'partial')[-4:],
                                      'gtcloud_path': gt_path})
                else:
                    file_list.append({
                        'taxonomy_id':
                            dc['taxonomy_id'],
                        'model_id':
                            s,
                        'partial_cloud_path': [
                            cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (subset, dc['taxonomy_id'], s, i)
                            for i in range(n_renderings)
                        ],
                        'gtcloud_path':
                            cfg.DATASETS.SHAPENET.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s),
                    })

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list


# ShapeNet-55/34
# Ref: https://github.com/yuxumin/PoinTr/blob/master/datasets/ShapeNet55Dataset.py
class ShapeNet55Dataset(torch.utils.data.dataset.Dataset):
    """
    ShapeNet55 dataset: return complete clouds, partial clouds are generated online
    """

    def __init__(self, options, file_list, transforms=None):
        self.options = options
        self.file_list = file_list
        self.transforms = transforms
        self.cache = dict()

    def __len__(self):
        return len(self.file_list)

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}

        for ri in self.options['required_items']:
            file_path = sample['%s_path' % ri]
            data[ri] = IO.get(file_path).astype(np.float32)
            # shapenet55
            data[ri] = self.pc_norm(data[ri])
            data[ri] = torch.from_numpy(data[ri]).float()

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], data


class ShapeNet55DataLoader(object):
    """
    ShapeNet55: get dataset file list
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = None
        return ShapeNet55Dataset(
            {
                'required_items': ['gtcloud'],
                'shuffle': subset == DatasetSubset.TRAIN
            }, file_list, transforms)

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""

        # Load the dataset indexing file
        with open(
                os.path.join(cfg.DATASETS.SHAPENET55.CATEGORY_FILE_PATH,
                             subset + '.txt'), 'r') as f:
            lines = f.readlines()

        # Collect file list
        file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            file_list.append({
                'taxonomy_id':
                    taxonomy_id,
                'model_id':
                    model_id,
                'gtcloud_path':
                    cfg.DATASETS.SHAPENET55.COMPLETE_POINTS_PATH % (line),
            })

        print('Complete collecting files of the dataset. Total files: %d' %
              len(file_list))
        return file_list


# Ref: https://github.com/yuxumin/PoinTr/blob/master/datasets/ShapeNet55Dataset.py
class ShapeNetW3d55Dataset(torch.utils.data.dataset.Dataset):
    """
    ShapeNet55 dataset: return complete clouds, partial clouds are generated online
    """

    def __init__(self, options, file_list, transforms=None):
        self.options = options
        self.file_list = file_list
        self.transforms = transforms
        self.cache = dict()
        self.view_types = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
        self.img_transforms = trans.Compose([
            # transforms.Scale(224),
            trans.Resize(224),
            trans.ToTensor()
        ])

    def __len__(self):
        return len(self.file_list)

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}

        for ri in self.options['required_items']:
            file_path = sample['%s_path' % ri]
            data[ri] = IO.get(file_path).astype(np.float32)
            # shapenet55
            data[ri] = self.pc_norm(data[ri])
            data[ri] = torch.from_numpy(data[ri]).float()

        if self.transforms is not None:
            data = self.transforms(data)

        if 'view_path' in sample:
            view_lst = []
            for i, view_type in enumerate(self.view_types):
                view_path = sample['view_path'] + 'depth_000_' + view_type + '.pfm'
                image_data, _ = self.read_pfm(view_path)
                views = self.img_transforms(Image.fromarray(image_data))
                # view_path = sample['view_path'] + 'depth_000_' + view_type + '.png'
                # image_data = np.clip(np.asarray(Image.open(view_path)) / 256.0, a_min=0.0, a_max=255.0)
                # views = self.img_transforms(Image.fromarray(np.uint8(image_data)))
                # views = views[:3, :, :]
                view_lst.append(views)

            data['views'] = torch.stack(view_lst).float()

        return sample['taxonomy_id'], sample['model_id'], data

    def read_pfm(self, path):
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


class ShapeNetW3d55DataLoader(object):
    """
    ShapeNet55: get dataset file list
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = None
        return ShapeNetW3d55Dataset(
            {
                'required_items': ['gtcloud'],
                'shuffle': subset == DatasetSubset.TRAIN
            }, file_list, transforms)

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""

        # Load the dataset indexing file
        with open(
                os.path.join(cfg.DATASETS.SHAPENET55.CATEGORY_FILE_PATH,
                             subset + '.txt'), 'r') as f:
            lines = f.readlines()

        # Collect file list
        file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            # model_id = line.split('-')[1].split('.')[0]
            model_id = line[9:-4]
            if model_id == '':
                if taxonomy_id == '04379243':
                    model_id = 'afda402f59a7737ad11ab08e7440'
                else:
                    model_id = '3a982b20a1c8ebf487b2ae2815c9'
            view_path = cfg.DATASETS.SHAPENET55.VIEW_PATH % (taxonomy_id, model_id)
            if not os.path.exists(view_path):
                print(taxonomy_id, model_id)
                continue
            file_list.append({
                'taxonomy_id':
                    taxonomy_id,
                'model_id':
                    model_id,
                'gtcloud_path':
                    cfg.DATASETS.SHAPENET55.COMPLETE_POINTS_PATH % (line),
                'view_path':
                    view_path
            })

        print('Complete collecting files of the dataset. Total files: %d' %
              len(file_list))
        return file_list


class ShapeNetSVD55Dataset(torch.utils.data.dataset.Dataset):
    """
    ShapeNet55 dataset: return complete clouds, partial clouds are generated online
    """

    def __init__(self, options, file_list, transforms=None):
        self.options = options
        self.file_list = file_list
        self.transforms = transforms
        self.cache = dict()
        self.view_ids = [0, 2, 5, 11, 16, 19]
        self.img_transforms = trans.Compose([
            # transforms.Scale(224),
            trans.Resize(224),
            trans.ToTensor()
        ])

    def __len__(self):
        return len(self.file_list)

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}

        for ri in self.options['required_items']:
            file_path = sample['%s_path' % ri]
            data[ri] = IO.get(file_path).astype(np.float32)
            # shapenet55
            data[ri] = self.pc_norm(data[ri])
            data[ri] = torch.from_numpy(data[ri]).float()

        if self.transforms is not None:
            data = self.transforms(data)

        if 'view_path' in sample:
            view_lst = []
            for i, view_type in enumerate(self.view_ids):
                view_path = sample['view_path'] + str(self.view_ids[i]).rjust(2, '0') + '.pfm'
                image_data, _ = self.read_pfm(view_path)
                views = self.img_transforms(Image.fromarray(image_data))
                # view_path = sample['view_path'] + 'depth_000_' + view_type + '.png'
                # image_data = np.clip(np.asarray(Image.open(view_path)) / 256.0, a_min=0.0, a_max=255.0)
                # views = self.img_transforms(Image.fromarray(np.uint8(image_data)))
                # views = views[:3, :, :]
                view_lst.append(views)

            data['views'] = torch.stack(view_lst).float()

        return sample['taxonomy_id'], sample['model_id'], data

    def read_pfm(self, path):
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


class ShapeNetSVD55DataLoader(object):
    """
    ShapeNet55: get dataset file list
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = None
        return ShapeNetSVD55Dataset(
            {
                'required_items': ['gtcloud'],
                'shuffle': subset == DatasetSubset.TRAIN
            }, file_list, transforms)

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""

        # Load the dataset indexing file
        with open(
                os.path.join(cfg.DATASETS.SHAPENET55.CATEGORY_FILE_PATH,
                             subset + '.txt'), 'r') as f:
            lines = f.readlines()

        # Collect file list
        file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            # model_id = line.split('-')[1].split('.')[0]
            model_id = line[9:-4]
            if model_id == '':
                if taxonomy_id == '04379243':
                    model_id = 'afda402f59a7737ad11ab08e7440'
                else:
                    model_id = '3a982b20a1c8ebf487b2ae2815c9'
            view_path = cfg.DATASETS.SHAPENET55.VIEW_PATH % (taxonomy_id, model_id)
            if not os.path.exists(view_path):
                print(taxonomy_id, model_id)
                continue
            file_list.append({
                'taxonomy_id':
                    taxonomy_id,
                'model_id':
                    model_id,
                'gtcloud_path':
                    cfg.DATASETS.SHAPENET55.COMPLETE_POINTS_PATH % (line),
                'view_path':
                    view_path
            })

        print('Complete collecting files of the dataset. Total files: %d' %
              len(file_list))
        return file_list


class ShapeNetW3dDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, options, file_list, transforms=None):
        self.options = options
        self.file_list = file_list
        self.transforms = transforms
        self.img_transforms = trans.Compose([
            # transforms.Scale(224),
            trans.Resize(224),
            trans.ToTensor()
        ])
        self.cache = dict()
        # self.view_ids = [0, 1, 2, 3, 4, 5]
        # self.view_types  = ['right', 'back', 'left', 'front_left', 'front', 'front_right']
        self.view_types = ['front', 'front_right', 'right', 'back', 'left', 'front_left']

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = -1
        if 'n_renderings' in self.options:
            rand_idx = random.randint(0, self.options['n_renderings'] - 1) if self.options['shuffle'] else 0

        for ri in self.options['required_items']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            data[ri] = IO.get(file_path).astype(np.float32)

        if self.transforms is not None:
            data = self.transforms(data)

        if 'view_path' in sample:
            view_lst = []
            for i, view_type in enumerate(self.view_types):
                view_path = sample['view_path'] + 'depth_000_' + view_type + '.pfm'
                image_data, _ = self.read_pfm(view_path)
                # mask = (image_data > 0)
                # image_data = (1 - image_data) * mask
                # image_data = 1 - image_data
                views = self.img_transforms(Image.fromarray(image_data))
                # view_path = sample['view_path'] + 'depth_000_' + view_type + '.png'
                # image_data = np.clip(np.asarray(Image.open(view_path)) / 256.0, a_min=0.0, a_max=255.0)
                # views = self.img_transforms(Image.fromarray(np.uint8(image_data)))
                # views = views[:3, :, :]
                view_lst.append(views)

            data['views'] = torch.stack(view_lst).float()

        return sample['taxonomy_id'], sample['model_id'], data

    def read_pfm(self, path):
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


class ShapeNetW3dDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.SHAPENET.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        num_views = self.cfg.DATASETS.SHAPENET.N_RENDERINGS
        n_renderings = self.cfg.DATASETS.SHAPENET.N_RENDERINGS if subset == DatasetSubset.TRAIN else 1
        file_list = self._get_file_list(self.cfg, self._get_subset(subset), n_renderings)
        transforms = self._get_transforms(self.cfg, subset)
        return ShapeNetW3dDataset({
            'n_renderings': n_renderings,
            'required_items': ['partial_cloud', 'gtcloud'] + [f'mv_cloud{i}' for i in range(num_views)],
            'shuffle': subset == DatasetSubset.TRAIN
        }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        num_views = self.cfg.DATASETS.SHAPENET.N_RENDERINGS
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
                },
                'objects': ['partial_cloud'] + [f'mv_cloud{i}' for i in range(num_views)]
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud'] + [f'mv_cloud{i}' for i in range(num_views)]
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
                },
                'objects': ['partial_cloud'] + [f'mv_cloud{i}' for i in range(num_views)]
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud'] + [f'mv_cloud{i}' for i in range(num_views)]
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []
        num_views = self.cfg.DATASETS.SHAPENET.N_RENDERINGS
        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):

                if subset == 'test':
                    partial_path = cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (dc['taxonomy_id'], s, 0)
                    # if dc['taxonomy_id'] == '02958343' or dc['taxonomy_id'] == '04530566':
                    #     continue
                    #     # partial_path = cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (dc['taxonomy_id'], s, 0)
                    # else:
                    #     partial_path = cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (dc['taxonomy_id'], s, 2)
                    gt_path = cfg.DATASETS.SHAPENET.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s)
                    view_path = cfg.DATASETS.SHAPENET.VIEW_PATH % (dc['taxonomy_id'], s)
                    if not os.path.exists(gt_path) or not os.path.exists(partial_path) or not os.path.exists(view_path):
                        continue
                    # if dc['taxonomy_id'] == '02958343' or dc['taxonomy_id'] == '04530566':
                    #     view_num_lst = [0, 1, 2, 4, 6, 7]
                    # else:
                    #     view_num_lst = [2, 3, 4, 6, 0, 1]
                    view_num_lst = [0, 1, 2, 4, 6, 7]
                    item = {'taxonomy_id': dc['taxonomy_id'],
                                      'model_id': s,
                                      'partial_cloud_path': partial_path,
                                      'gtcloud_path': gt_path,
                                      'view_path': view_path}
                    item.update(
                        {f'mv_cloud{i}_path': cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (dc['taxonomy_id'], s, view_num)
                         for i, view_num in enumerate(view_num_lst)})
                    file_list.append(item)
                else:
                    partial_path = cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (dc['taxonomy_id'], s, 0)
                    # if dc['taxonomy_id'] == '02958343' or dc['taxonomy_id'] == '04530566':
                    #     continue
                    #     # partial_path = cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (dc['taxonomy_id'], s, 0)
                    # else:
                    #     partial_path = cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (dc['taxonomy_id'], s, 2)
                    gt_path = cfg.DATASETS.SHAPENET.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s)
                    view_path = cfg.DATASETS.SHAPENET.VIEW_PATH % (dc['taxonomy_id'], s)
                    if not os.path.exists(gt_path) or not os.path.exists(partial_path) or not os.path.exists(view_path):
                        continue
                    view_num_lst = [0, 1, 2, 4, 6, 7]
                    # if dc['taxonomy_id'] == '02958343' or dc['taxonomy_id'] == '04530566':
                    #     view_num_lst = [0, 1, 2, 4, 6, 7]
                    # else:
                    #     view_num_lst = [2, 3, 4, 6, 0, 1]
                    item = {
                        'taxonomy_id':
                            dc['taxonomy_id'],
                        'model_id':
                            s,
                        'partial_cloud_path': partial_path,
                        'gtcloud_path': gt_path,
                        'view_path': view_path
                    }
                    item.update(
                        {f'mv_cloud{i}_path': cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (dc['taxonomy_id'], s, view_num)
                         for i, view_num in enumerate(view_num_lst)})
                    file_list.append(item)

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list


class ShapeNetSVDDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, options, file_list, transforms=None):
        self.options = options
        self.file_list = file_list
        self.transforms = transforms
        self.img_transforms = trans.Compose([
            # transforms.Scale(224),
            trans.Resize(224),
            trans.ToTensor()
        ])
        self.cache = dict()
        self.view_ids = [0, 2, 5, 11, 16, 19]
        # self.view_types  = ['right', 'back', 'left', 'front_left', 'front', 'front_right']
        # self.view_types = ['front', 'front_right', 'right', 'back', 'left', 'front_left']

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = -1
        if 'n_renderings' in self.options:
            rand_idx = random.randint(0, self.options['n_renderings'] - 1) if self.options['shuffle'] else 0

        for ri in self.options['required_items']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            data[ri] = IO.get(file_path).astype(np.float32)

        if self.transforms is not None:
            data = self.transforms(data)

        if 'view_path' in sample:
            view_lst = []
            for i, view_type in enumerate(self.view_ids):
                view_path = sample['view_path'] + str(self.view_ids[i]).rjust(2, '0') + '.pfm'
                image_data, _ = self.read_pfm(view_path)
                views = self.img_transforms(Image.fromarray(image_data))
                # view_path = sample['view_path'] + 'depth_000_' + view_type + '.png'
                # image_data = np.clip(np.asarray(Image.open(view_path)) / 256.0, a_min=0.0, a_max=255.0)
                # views = self.img_transforms(Image.fromarray(np.uint8(image_data)))
                # views = views[:3, :, :]
                view_lst.append(views)

            data['views'] = torch.stack(view_lst).float()

        return sample['taxonomy_id'], sample['model_id'], data

    def read_pfm(self, path):
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


class ShapeNetSVDDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.SHAPENET.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        num_views = self.cfg.DATASETS.SHAPENET.N_RENDERINGS
        n_renderings = self.cfg.DATASETS.SHAPENET.N_RENDERINGS if subset == DatasetSubset.TRAIN else 1
        file_list = self._get_file_list(self.cfg, self._get_subset(subset), n_renderings)
        transforms = self._get_transforms(self.cfg, subset)
        return ShapeNetSVDDataset({
            'n_renderings': n_renderings,
            'required_items': ['partial_cloud', 'gtcloud'],
            'shuffle': subset == DatasetSubset.TRAIN
        }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        num_views = self.cfg.DATASETS.SHAPENET.N_RENDERINGS
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []
        num_views = self.cfg.DATASETS.SHAPENET.N_RENDERINGS
        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):

                if subset == 'test':
                    partial_path = cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (dc['taxonomy_id'], s, 0)
                    # if dc['taxonomy_id'] == '02958343' or dc['taxonomy_id'] == '04530566':
                    #     continue
                    #     # partial_path = cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (dc['taxonomy_id'], s, 0)
                    # else:
                    #     partial_path = cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (dc['taxonomy_id'], s, 2)
                    gt_path = cfg.DATASETS.SHAPENET.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s)
                    view_path = cfg.DATASETS.SHAPENET.VIEW_PATH % (dc['taxonomy_id'], s)
                    if not os.path.exists(gt_path) or not os.path.exists(partial_path) or not os.path.exists(view_path):
                        continue
                    # if dc['taxonomy_id'] == '02958343' or dc['taxonomy_id'] == '04530566':
                    #     view_num_lst = [0, 1, 2, 4, 6, 7]
                    # else:
                    #     view_num_lst = [2, 3, 4, 6, 0, 1]
                    # view_num_lst = [0, 1, 2, 4, 6, 7]
                    item = {'taxonomy_id': dc['taxonomy_id'],
                                      'model_id': s,
                                      'partial_cloud_path': partial_path,
                                      'gtcloud_path': gt_path,
                                      'view_path': view_path}
                    file_list.append(item)
                else:
                    partial_path = cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (dc['taxonomy_id'], s, 0)
                    # if dc['taxonomy_id'] == '02958343' or dc['taxonomy_id'] == '04530566':
                    #     continue
                    #     # partial_path = cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (dc['taxonomy_id'], s, 0)
                    # else:
                    #     partial_path = cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (dc['taxonomy_id'], s, 2)
                    gt_path = cfg.DATASETS.SHAPENET.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s)
                    view_path = cfg.DATASETS.SHAPENET.VIEW_PATH % (dc['taxonomy_id'], s)
                    if not os.path.exists(gt_path) or not os.path.exists(partial_path) or not os.path.exists(view_path):
                        continue
                    # view_num_lst = [0, 1, 2, 4, 6, 7]
                    # if dc['taxonomy_id'] == '02958343' or dc['taxonomy_id'] == '04530566':
                    #     view_num_lst = [0, 1, 2, 4, 6, 7]
                    # else:
                    #     view_num_lst = [2, 3, 4, 6, 0, 1]
                    item = {
                        'taxonomy_id':
                            dc['taxonomy_id'],
                        'model_id':
                            s,
                        'partial_cloud_path': partial_path,
                        'gtcloud_path': gt_path,
                        'view_path': view_path
                    }
                    file_list.append(item)

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list



# //////////////////////////////////////////// = Dataset Loader Mapping = //////////////////////////////////////////// #

DATASET_LOADER_MAPPING = {
    'ShapeNet': ShapeNetDataLoader,
    'ShapeNet55': ShapeNet55DataLoader,
    'ShapeNetW3d55': ShapeNetW3d55DataLoader,
    'ShapeNetSVD55': ShapeNetSVD55DataLoader,
    'ShapeNetW3d': ShapeNetW3dDataLoader,
    'ShapeNetSVD': ShapeNetSVDDataLoader,
}  # yapf: disable
