import os
import copy
import h5py
import torch
import pickle
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset
from os.path import join
import cv2
from numpy.linalg import inv
from .utils import scale_intrinsics, warp_depth, warp_points2d

def split(n, k):
    d, r = divmod(n, k)
    return [d + 1] * r + [d] * (k - r)

def read_scannet_pose(path):
    """ Read ScanNet's Camera2World pose and transform it to World2Camera.
    
    Returns:
        pose_w2c (np.ndarray): (4, 4)
    """
    cam2world = np.loadtxt(path, delimiter=' ')
    world2cam = inv(cam2world)
    return world2cam

class ScanNetDataset(Dataset):
    def __init__(
            self,
            root,
            npz_path,
            seq_name,
            num_per_scene=100,
            image_size=256,
            min_overlap_score=0.1,
            max_overlap_score=0.9,
            gray=False,
            crop_or_scale='scale',  # crop, scale, crop_scale
            train=True,
            seed=42,  # seed parameter, default 42
    ):
        # Read paths under sub-scene
        self.data_path = join(root, seq_name)
        npz_path = join(npz_path, seq_name + '.npz')

        # Dataset parameter setup
        self.num_per_scene = num_per_scene
        self.train = train
        self.image_size = image_size
        self.gray = gray
        self.crop_or_scale = crop_or_scale
        self.seed = seed  # save seed for controlling randomness

        # Load scene info and pair data (following MegaDepth scene_info structure)
        self.scene_info = dict(np.load(npz_path, allow_pickle=True))
        self.pair_infos = []  # pair info storage format consistent with MegaDepth

        # Parse pair data
        pair_ids = self.scene_info['name']  # (n, 4)
        pair_scr = self.scene_info['score']  # (n,)

        # Filter valid pairs
        valid = (pair_ids[:, -2:] % 10).sum(axis=-1) == 0
        pair_ids = pair_ids[valid]
        pair_scr = pair_scr[valid]

        # Sample by overlap score intervals
        intervals = np.concatenate((
            np.arange(min_overlap_score, max_overlap_score, 0.1),
            [max_overlap_score]
        ))
        sub_max_samples = split(1000, len(intervals) - 1)

        # Set random module seed to control random.sample behavior
        random.seed(self.seed)

        for low, upr, m in zip(intervals[:-1], intervals[1:], sub_max_samples):
            valid = (pair_scr > low) & (pair_scr <= upr)
            ids = pair_ids[valid]
            scr = pair_scr[valid]

            if len(ids) > m:
                samples = random.sample(list(range(len(ids))), m)
                ids = ids[samples]
                scr = scr[samples]

            # Convert to pair_infos format consistent with MegaDepth: [( (id0, id1), score ), ...]
            for idx in range(len(ids)):
                scene_name, sub_name, id0, id1 = ids[idx]
                self.pair_infos.append( ( (id0, id1), scr[idx], scene_name, sub_name ) )

        # Limit samples per scene
        if len(self.pair_infos) > num_per_scene:
            # Set numpy seed to control np.random.choice behavior
            np.random.seed(self.seed)
            indices = np.random.choice(len(self.pair_infos), num_per_scene, replace=False)
            self.pair_infos = [self.pair_infos[idx] for idx in indices]

        # Image transforms consistent with MegaDepth
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        # Load intrinsics (following MegaDepth scene_info storage)
        intrinsic_path = "/high_perf_store3/world-model/liuyepeng/data/open_source/scannet/scannet_indices/train_scene_info/intrinsics.npz"
        intrinsics_params = dict(np.load(intrinsic_path))
        scene_name_ = f'scene{self.pair_infos[0][-2]:04d}_{self.pair_infos[0][-1]:02d}'
        # Get NumPy intrinsic matrix and reshape to 3x3
        K0 = intrinsics_params[scene_name_].copy().reshape(3, 3)  # use copy to avoid modifying original data

        # Broadcast to generate a list of K0 copies with same length as pair_infos
        self.scene_info['intrinsics'] = [K0.copy() for _ in range(len(self.pair_infos))]
       
            
    def __len__(self):
        return len(self.pair_infos)

    def recover_pair(self, idx):
        # scannet has no central match point labels
        (idx0, idx1), overlap_score, scene_name,  sub_name= self.pair_infos[idx % len(self)]

        # Read image paths
        image_path1 = join(self.data_path, 'color', f'{idx0}.jpg')
        image_path2 = join(self.data_path, 'color', f'{idx1}.jpg')
        depth_path1 = join(self.data_path, 'depth', f'{idx0}.png')
        depth_path2 = join(self.data_path, 'depth', f'{idx1}.png')
        pose_path1 = join(self.data_path, 'pose', f'{idx0}.txt')
        pose_path2 = join(self.data_path, 'pose', f'{idx1}.txt')

        # Read images
        image1 = Image.open(image_path1)
        if image1.mode != 'RGB':
            image1 = image1.convert('RGB')
        image1 = np.array(image1) # （h, w, c)
        image1 = cv2.resize(image1, (640, 480))
        depth1 = cv2.imread(depth_path1, cv2.IMREAD_UNCHANGED) / 1000
        assert (image1.shape[0] == depth1.shape[0] and image1.shape[1] == depth1.shape[1])
        intrinsics1 = self.scene_info['intrinsics'][0]
        pose1 = read_scannet_pose(pose_path1)

        image2 = Image.open(image_path2)
        if image2.mode != 'RGB':
            image2 = image2.convert('RGB')
        image2 = np.array(image2)
        image2 = cv2.resize(image2, (640, 480))
        depth2 = cv2.imread(depth_path2, cv2.IMREAD_UNCHANGED) / 1000
        assert (image2.shape[0] == depth2.shape[0] and image2.shape[1] == depth2.shape[1])
        intrinsics2 = self.scene_info['intrinsics'][0]
        pose2 = read_scannet_pose(pose_path2)

        pose12 = pose2 @ np.linalg.inv(pose1)
        pose21 = np.linalg.inv(pose12)

        if self.train:
            if "crop" in self.crop_or_scale:
                # ================================================= compute central_match
                DOWNSAMPLE = 10
                # resize to speed up
                depth1s = cv2.resize(depth1, (depth1.shape[1] // DOWNSAMPLE, depth1.shape[0] // DOWNSAMPLE))
                depth2s = cv2.resize(depth2, (depth2.shape[1] // DOWNSAMPLE, depth2.shape[0] // DOWNSAMPLE))
                intrinsic1s = scale_intrinsics(intrinsics1, (DOWNSAMPLE, DOWNSAMPLE))
                intrinsic2s = scale_intrinsics(intrinsics2, (DOWNSAMPLE, DOWNSAMPLE))

                # warp
                depth12s = warp_depth(depth1s, intrinsic1s, intrinsic2s, pose12, depth2s.shape)
                depth21s = warp_depth(depth2s, intrinsic2s, intrinsic1s, pose21, depth1s.shape)

                depth12s[depth12s < 0] = 0
                depth21s[depth21s < 0] = 0

                valid12s = np.logical_and(depth12s > 0, depth2s > 0)
                valid21s = np.logical_and(depth21s > 0, depth1s > 0)

                pos1 = np.array(valid21s.nonzero())
                try:
                    idx1_random = np.random.choice(np.arange(pos1.shape[1]), 1)
                    uv1s = pos1[:, idx1_random][[1, 0]].reshape(1, 2)
                    d1s = np.array(depth1s[uv1s[0, 1], uv1s[0, 0]]).reshape(1, 1)

                    uv12s, z12s = warp_points2d(uv1s, d1s, intrinsic1s, intrinsic2s, pose12)

                    uv1 = uv1s[0] * DOWNSAMPLE
                    uv2 = uv12s[0] * DOWNSAMPLE
                except ValueError:
                    uv1 = [depth1.shape[1] / 2, depth1.shape[0] / 2]
                    uv2 = [depth2.shape[1] / 2, depth2.shape[0] / 2]

                central_match = [uv1[1], uv1[0], uv2[1], uv2[0]]
                # ================================================= compute central_match

            if self.crop_or_scale == 'crop':
                # =============== padding
                h1, w1, _ = image1.shape
                h2, w2, _ = image2.shape
                if h1 < self.image_size:
                    padding = np.zeros((self.image_size - h1, w1, 3))
                    image1 = np.concatenate([image1, padding], axis=0).astype(np.uint8)
                    depth1 = np.concatenate([depth1, padding[:, :, 0]], axis=0).astype(np.float32)
                    h1, w1, _ = image1.shape
                if w1 < self.image_size:
                    padding = np.zeros((h1, self.image_size - w1, 3))
                    image1 = np.concatenate([image1, padding], axis=1).astype(np.uint8)
                    depth1 = np.concatenate([depth1, padding[:, :, 0]], axis=1).astype(np.float32)
                if h2 < self.image_size:
                    padding = np.zeros((self.image_size - h2, w2, 3))
                    image2 = np.concatenate([image2, padding], axis=0).astype(np.uint8)
                    depth2 = np.concatenate([depth2, padding[:, :, 0]], axis=0).astype(np.float32)
                    h2, w2, _ = image2.shape
                if w2 < self.image_size:
                    padding = np.zeros((h2, self.image_size - w2, 3))
                    image2 = np.concatenate([image2, padding], axis=1).astype(np.uint8)
                    depth2 = np.concatenate([depth2, padding[:, :, 0]], axis=1).astype(np.float32)
                # =============== padding
                image1, bbox1, image2, bbox2 = self.crop(image1, image2, central_match)

                depth1 = depth1[bbox1[0]: bbox1[0] + self.image_size, bbox1[1]: bbox1[1] + self.image_size]
                depth2 = depth2[bbox2[0]: bbox2[0] + self.image_size, bbox2[1]: bbox2[1] + self.image_size]
            elif self.crop_or_scale == 'scale':
                image1, depth1, intrinsics1 = self.scale(image1, depth1, intrinsics1)
                image2, depth2, intrinsics2 = self.scale(image2, depth2, intrinsics2)
                bbox1 = bbox2 = np.array([0., 0.])
            elif self.crop_or_scale == 'crop_scale':
                bbox1 = bbox2 = np.array([0., 0.])
                image1, depth1, intrinsics1 = self.crop_scale(image1, depth1, intrinsics1, central_match[:2])
                image2, depth2, intrinsics2 = self.crop_scale(image2, depth2, intrinsics2, central_match[2:])
            else:
                raise RuntimeError(f"Unkown type {self.crop_or_scale}")
        else:
            bbox1 = bbox2 = np.array([0., 0.])
        return (image1, depth1, intrinsics1, pose12, bbox1,
                image2, depth2, intrinsics2, pose21, bbox2)

    def scale(self, image, depth, intrinsic):
        img_size_org = image.shape
        image = cv2.resize(image, (self.image_size, self.image_size))
        depth = cv2.resize(depth, (self.image_size, self.image_size))
        intrinsic = scale_intrinsics(intrinsic, (img_size_org[1] / self.image_size, img_size_org[0] / self.image_size))
        return image, depth, intrinsic

    def crop_scale(self, image, depth, intrinsic, centeral):
        h_org, w_org, three = image.shape
        image_size = min(h_org, w_org)
        if h_org > w_org:
            if centeral[1] - image_size // 2 < 0:
                h_start = 0
            elif centeral[1] + image_size // 2 > h_org:
                h_start = h_org - image_size
            else:
                h_start = int(centeral[1]) - image_size // 2
            w_start = 0
        else:
            if centeral[0] - image_size // 2 < 0:
                w_start = 0
            elif centeral[0] + image_size // 2 > w_org:
                w_start = w_org - image_size
            else:
                w_start = int(centeral[0]) - image_size // 2
            h_start = 0

        croped_image = image[h_start: h_start + image_size, w_start: w_start + image_size]
        croped_depth = depth[h_start: h_start + image_size, w_start: w_start + image_size]
        intrinsic[0, 2] = intrinsic[0, 2] - w_start
        intrinsic[1, 2] = intrinsic[1, 2] - h_start

        image = cv2.resize(croped_image, (self.image_size, self.image_size))
        depth = cv2.resize(croped_depth, (self.image_size, self.image_size))
        intrinsic = scale_intrinsics(intrinsic, (image_size / self.image_size, image_size / self.image_size))

        return image, depth, intrinsic

    def crop(self, image1, image2, central_match):
        bbox1_i = max(int(central_match[0]) - self.image_size // 2, 0)
        if bbox1_i + self.image_size >= image1.shape[0]:
            bbox1_i = image1.shape[0] - self.image_size
        bbox1_j = max(int(central_match[1]) - self.image_size // 2, 0)
        if bbox1_j + self.image_size >= image1.shape[1]:
            bbox1_j = image1.shape[1] - self.image_size

        bbox2_i = max(int(central_match[2]) - self.image_size // 2, 0)
        if bbox2_i + self.image_size >= image2.shape[0]:
            bbox2_i = image2.shape[0] - self.image_size
        bbox2_j = max(int(central_match[3]) - self.image_size // 2, 0)
        if bbox2_j + self.image_size >= image2.shape[1]:
            bbox2_j = image2.shape[1] - self.image_size

        return (image1[bbox1_i: bbox1_i + self.image_size, bbox1_j: bbox1_j + self.image_size],
                np.array([bbox1_i, bbox1_j]),
                image2[bbox2_i: bbox2_i + self.image_size, bbox2_j: bbox2_j + self.image_size],
                np.array([bbox2_i, bbox2_j])
                )

    def __getitem__(self, idx):
        
        (image1, depth1, intrinsics1, pose12, bbox1,
         image2, depth2, intrinsics2, pose21, bbox2) \
            = self.recover_pair(idx)
        
        if self.gray:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
            gray1 = transforms.ToTensor()(gray1)
            gray2 = transforms.ToTensor()(gray2)
        if self.transforms is not None:
            image1, image2 = self.transforms(image1), self.transforms(image2)  # [C,H,W]
        ret = {'image0': image1,
               'image1': image2,
               'angle': 0,
               'overlap': self.pair_infos[idx][1],
               'warp01_params': {'mode': 'se3',
                                 'width': self.image_size if self.train else image1.shape[2],
                                 'height': self.image_size if self.train else image1.shape[1],
                                 'pose01': torch.from_numpy(pose12.astype(np.float32)),
                                 'bbox0': torch.from_numpy(bbox1.astype(np.float32)),
                                 'bbox1': torch.from_numpy(bbox2.astype(np.float32)),
                                 'depth0': torch.from_numpy(depth1.astype(np.float32)),
                                 'depth1': torch.from_numpy(depth2.astype(np.float32)),
                                 'intrinsics0': torch.from_numpy(intrinsics1.astype(np.float32)),
                                 'intrinsics1': torch.from_numpy(intrinsics2.astype(np.float32))},
               'warp10_params': {'mode': 'se3',
                                 'width': self.image_size if self.train else image2.shape[2],
                                 'height': self.image_size if self.train else image2.shape[2],
                                 'pose01': torch.from_numpy(pose21.astype(np.float32)),
                                 'bbox0': torch.from_numpy(bbox2.astype(np.float32)),
                                 'bbox1': torch.from_numpy(bbox1.astype(np.float32)),
                                 'depth0': torch.from_numpy(depth2.astype(np.float32)),
                                 'depth1': torch.from_numpy(depth1.astype(np.float32)),
                                 'intrinsics0': torch.from_numpy(intrinsics2.astype(np.float32)),
                                 'intrinsics1': torch.from_numpy(intrinsics1.astype(np.float32))},
               }
        if self.gray:
            ret['gray0'] = gray1
            ret['gray1'] = gray2
        return ret


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt


    def visualize(image0, image1, depth0, depth1):
        # visualize image and depth
        plt.figure(figsize=(9, 9))
        plt.subplot(2, 2, 1)
        plt.imshow(image0, cmap='gray')
        plt.subplot(2, 2, 2)
        plt.imshow(depth0)
        plt.subplot(2, 2, 3)
        plt.imshow(image1, cmap='gray')
        plt.subplot(2, 2, 4)
        plt.imshow(depth1)
        plt.show()


    dataset = MegaDepthDataset(  # root='../data/megadepth',
        root='../data/imw2020val',
        train=False,
        using_cache=True,
        pairs_per_scene=100,
        image_size=256,
        colorjit=True,
        gray=False,
        crop_or_scale='scale',
    )
    dataset.build_dataset()

    batch_size = 2

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for idx, batch in enumerate(tqdm(loader)):
        image0, image1 = batch['image0'], batch['image1']  # [B,3,H,W]
        depth0, depth1 = batch['warp01_params']['depth0'], batch['warp01_params']['depth1']  # [B,H,W]
        intrinsics0, intrinsics1 = batch['warp01_params']['intrinsics0'], batch['warp01_params'][
            'intrinsics1']  # [B,3,3]

        batch_size, channels, h, w = image0.shape

        for b_idx in range(batch_size):
            visualize(image0[b_idx].permute(1, 2, 0), image1[b_idx].permute(1, 2, 0), depth0[b_idx], depth1[b_idx])