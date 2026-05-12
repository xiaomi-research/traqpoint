from .utils import scale_intrinsics, warp_depth, warp_points2d
import torch
import numpy as np
import cv2
import h5py
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

class MegaDepthSequenceDataset(Dataset):
    def __init__(
            self,
            root,
            npz_path,
            image_size=256,
            seq_len=5,
            num_per_scene=100,
            gray=False,
            crop_or_scale='scale',
            train=True,
            seed=42,
    ):
        self.data_path = Path(root)
        self.image_size = image_size
        self.seq_len = seq_len
        self.gray = gray
        self.crop_or_scale = crop_or_scale
        self.train = train
        self.seed = seed

        # Load npz file
        # npz_data = np.load(npz_path, allow_pickle=True)
        # self.scene_info = dict(npz_data['scene_info'].item())  # dict
        self.scene_info = dict(np.load(npz_path, allow_pickle=True))
        sequence_infos = self.scene_info['sequence_infos'].copy()  # [num_seq, 5]
        sequence_scores = self.scene_info['sequence_scores'].copy()  # [num_seq, 4]

        # ---------------------- New: Rotation parameter configuration ----------------------
        if self.crop_or_scale == 'scale_rotate':
            self.rotate_angles = [0, 90, 180, 270]  # Supported rotation angles
            self.rotate_probs = [0.4, 0.25, 0.1, 0.25]  # Corresponding probabilities
            # Bind rotation angles with cv2 rotation enums (ensure correct rotation methods)
            self.angle2cv_rotate = {
                0: None,  # No rotation
                90: cv2.ROTATE_90_CLOCKWISE,  # Clockwise 90 degrees
                180: cv2.ROTATE_180,  # 180 degrees
                270: cv2.ROTATE_90_COUNTERCLOCKWISE  # Counter-clockwise 90 degrees (i.e., clockwise 270 degrees)
            }


        # Simple quantity limit
        if len(sequence_infos) > num_per_scene:
            np.random.seed(self.seed)
            idxs = np.random.choice(len(sequence_infos), num_per_scene, replace=False)
            self.sequence_infos = sequence_infos[idxs]
            self.sequence_scores = sequence_scores[idxs]
        else:
            self.sequence_infos = sequence_infos
            self.sequence_scores = sequence_scores
        del self.scene_info['sequence_infos']
        del self.scene_info['sequence_scores']
        self.transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

    def __len__(self):
        return len(self.sequence_infos)

    def _get_frame_data(self, idx):
        img_name = self.scene_info['image_paths'][idx]
        depth_path = self.scene_info['depth_paths'][idx].replace(
            'phoenix/S6/zl548/MegaDepth_v1', 'depth_undistorted'
        )
        depth_path = '/'.join([depth_path.split('/')[i] for i in [0, 1, -1]])
        depth_full_path = self.data_path / depth_path
        with h5py.File(depth_full_path, 'r') as hdf5_file:
            depth = np.array(hdf5_file['/depth'])
        assert (np.min(depth) >= 0)
        
        image_path = self.data_path / img_name
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)
        assert (image.shape[0] == depth.shape[0] and image.shape[1] == depth.shape[1])
        intrinsics = self.scene_info['intrinsics'][idx].copy()
        pose = self.scene_info['poses'][idx]
        frame_id = idx
        return image, depth, intrinsics, pose, frame_id

    def scale(self, image, depth, intrinsic):
        img_size_org = image.shape
        image = cv2.resize(image, (self.image_size, self.image_size))
        depth = cv2.resize(depth, (self.image_size, self.image_size))
        intrinsic = scale_intrinsics(
            intrinsic, (img_size_org[1] / self.image_size, img_size_org[0] / self.image_size)
        )
        return image, depth, intrinsic


    def crop(self, image, depth, central_match):
        bbox_i = max(int(central_match[0]) - self.image_size // 2, 0)
        if bbox_i + self.image_size >= image.shape[0]:
            bbox_i = image.shape[0] - self.image_size
        bbox_j = max(int(central_match[1]) - self.image_size // 2, 0)
        if bbox_j + self.image_size >= image.shape[1]:
            bbox_j = image.shape[1] - self.image_size

        image_cropped = image[bbox_i: bbox_i + self.image_size, bbox_j: bbox_j + self.image_size]
        depth_cropped = depth[bbox_i: bbox_i + self.image_size, bbox_j: bbox_j + self.image_size]
        bbox = np.array([bbox_i, bbox_j])
        return image_cropped, depth_cropped, bbox

    def _adjust_intrinsic(self, intrinsic, bbox):
        """Adjust intrinsic parameters based on crop offset"""
        adjusted = intrinsic.copy()
        bbox_i, bbox_j = bbox  # Offset in height and width directions
        adjusted[0, 2] -= bbox_j  # Adjust cx
        adjusted[1, 2] -= bbox_i  # Adjust cy
        return adjusted

    # ---------------------- New: Reference frame intrinsic adjustment after rotation (Core!) ----------------------
    def _adjust_intrinsic_for_rotation(self, intrinsic, angle, img_size):
        """
        Adjust reference frame intrinsic parameters based on rotation angle to ensure 3D point projection consistency
        Args:
            intrinsic: Scaled reference frame intrinsic parameters (3x3 matrix)
            angle: Rotation angle (0/90/180/270)
            img_size: Rotated image size (S x S, S=self.image_size)
        Returns:
            adjusted_intrinsic: Rotation-adapted intrinsic parameters (3x3 matrix)
        """
        S = img_size[0]  # Image size (S x S, square)
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        adjusted = intrinsic.copy()

        if angle == 0:
            # No rotation: intrinsic parameters unchanged
            pass
        elif angle == 90:
            # Clockwise 90 degrees: width and height directions swap, principal point position adapts
            adjusted[0, 0] = fy          # New fx = original fy (width direction becomes original height direction)
            adjusted[1, 1] = fx          # New fy = original fx (height direction becomes original width direction)
            adjusted[0, 2] = cy          # New cx = original cy (width principal point corresponds to original height principal point)
            adjusted[1, 2] = (S - 1) - cx# New cy = image height-1 - original cx (height principal point reverses)
        elif angle == 180:
            # 180 degrees: width and height directions unchanged, principal point position reverses
            adjusted[0, 2] = (S - 1) - cx# New cx = image width-1 - original cx
            adjusted[1, 2] = (S - 1) - cy# New cy = image height-1 - original cy
        elif angle == 270:
            # Clockwise 270 degrees (counter-clockwise 90 degrees): width and height directions swap, principal point position adapts
            adjusted[0, 0] = fy          # New fx = original fy
            adjusted[1, 1] = fx          # New fy = original fx
            adjusted[0, 2] = (S - 1) - cy# New cx = image width-1 - original cy
            adjusted[1, 2] = cx          # New cy = original cx
        return adjusted

    def _get_rotation_matrix(self, angle):
        """Generate rotation matrix around z-axis (right-hand coordinate system, angle corresponds to image rotation direction)"""
        rad = np.radians(angle)
        if angle == 0:
            return np.eye(3)
        elif angle == 90:  # Clockwise 90 degrees → rotation matrix is counter-clockwise 90 degrees (right-hand system)
            return np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        elif angle == 180:
            return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        elif angle == 270:  # Clockwise 270 degrees → rotation matrix is counter-clockwise 270 degrees (i.e., clockwise 90 degrees)
            return np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    def __getitem__(self, idx):
        frame_indices = self.sequence_infos[idx]  # [5]
        scores = self.sequence_scores[idx]        # [4]
        ref_idx = 0
        # Sample center point for cropping/alignment (can use reference frame center or other strategies)
       
        # Store original data for all frames (for subsequent processing)
        original_data = []
        for frame_idx in frame_indices:
            image, depth, intrinsic, pose, frame_id = self._get_frame_data(frame_idx)
            original_data.append((image, depth, intrinsic, pose, frame_id))

        # Check reference frame original center depth validity (depth before scaling is more reliable)
        ref_image, ref_depth, ref_intrinsics, ref_pose, ref_frame_id = original_data[ref_idx]
        ref_original_h, ref_original_w = ref_image.shape[:2]
        ref_original_center_v, ref_original_center_u = ref_original_h//2, ref_original_w//2
        ref_original_center_depth = ref_depth[ref_original_center_v, ref_original_center_u]
        depth_valid = (ref_original_center_depth > 0)

        # Initialize storage lists
        images, depths, intrinsics, poses, frame_ids, bboxes, processed_images = [], [], [], [], [], [], []
        if self.crop_or_scale == 'crop':
            if depth_valid:
                # Branch 1: Reference frame depth valid → reference frame scale + other frames projection cropping
                # Process reference frame (scale)
                ref_image_scaled, ref_depth_scaled, ref_intrinsic_scaled = self.scale(
                    ref_image, ref_depth, ref_intrinsics
                )
                ref_bbox = np.array([0., 0.])
                images.append(ref_image_scaled)
                depths.append(ref_depth_scaled)
                intrinsics.append(ref_intrinsic_scaled)
                poses.append(ref_pose)
                frame_ids.append(ref_frame_id)
                bboxes.append(ref_bbox)
                processed_images.append(self.transforms(ref_image_scaled))

                # Calculate reference frame scaled center 3D coordinates
                ref_scaled_h, ref_scaled_w = ref_image_scaled.shape[:2]
                ref_scaled_center_u, ref_scaled_center_v = ref_scaled_w//2, ref_scaled_h//2
                fx, fy = ref_intrinsic_scaled[0, 0], ref_intrinsic_scaled[1, 1]
                cx, cy = ref_intrinsic_scaled[0, 2], ref_intrinsic_scaled[1, 2]
                x = (ref_scaled_center_u - cx) * ref_original_center_depth / fx
                y = (ref_scaled_center_v - cy) * ref_original_center_depth / fy
                z = ref_original_center_depth
                ref_center_3d = np.array([x, y, z, 1.0])  # Homogeneous coordinates

                # Process other frames (projection cropping)
                for i in range(1, self.seq_len):
                    img, dpt, K, pose, fid = original_data[i]
                    h, w = img.shape[:2]

                    
                    if h < self.image_size:
                        pad_h = self.image_size - h
                        img = np.pad(img, ((0, pad_h), (0, 0), (0, 0)), mode='constant', constant_values=0).astype(img.dtype)
                        dpt = np.pad(dpt, ((0, pad_h), (0, 0)), mode='constant', constant_values=0).astype(dpt.dtype)
                    if w < self.image_size:
                        pad_w = self.image_size - w
                        img = np.pad(img, ((0, 0), (0, pad_w), (0, 0)), mode='constant', constant_values=0).astype(img.dtype)
                        dpt = np.pad(dpt, ((0, 0), (0, pad_w)), mode='constant', constant_values=0).astype(dpt.dtype)

                    # Project reference frame 3D center to current frame
                    rel_pose = pose @ np.linalg.inv(ref_pose)
                    curr_center_3d = rel_pose @ ref_center_3d
                    curr_center_3d = curr_center_3d[:3] / curr_center_3d[3]

                    # Calculate current frame cropping center
                    fx_curr, fy_curr = K[0, 0], K[1, 1]
                    cx_curr, cy_curr = K[0, 2], K[1, 2]
                    curr_u = (fx_curr * curr_center_3d[0] / curr_center_3d[2]) + cx_curr
                    curr_v = (fy_curr * curr_center_3d[1] / curr_center_3d[2]) + cy_curr
                    curr_crop_center = [curr_v, curr_u]

                    
                    img_cropped, dpt_cropped, bbox = self.crop(img, dpt, curr_crop_center)
                    K_adjusted = self._adjust_intrinsic(K, bbox)

                    
                    images.append(img_cropped)
                    depths.append(dpt_cropped)
                    intrinsics.append(K_adjusted)
                    poses.append(pose)
                    frame_ids.append(fid)
                    bboxes.append(bbox)
                    processed_images.append(self.transforms(img_cropped))
            else:
                # Branch 2: Reference frame depth invalid → all frames crop using their own center
                for i in range(self.seq_len):
                    img, dpt, K, pose, fid = original_data[i]
                    h, w = img.shape[:2]

                   
                    if h < self.image_size:
                        pad_h = self.image_size - h
                        img = np.pad(img, ((0, pad_h), (0, 0), (0, 0)), mode='constant', constant_values=0).astype(img.dtype)
                        dpt = np.pad(dpt, ((0, pad_h), (0, 0)), mode='constant', constant_values=0).astype(dpt.dtype)
                    if w < self.image_size:
                        pad_w = self.image_size - w
                        img = np.pad(img, ((0, 0), (0, pad_w), (0, 0)), mode='constant', constant_values=0).astype(img.dtype)
                        dpt = np.pad(dpt, ((0, 0), (0, pad_w)), mode='constant', constant_values=0).astype(dpt.dtype)

                    # Use own center as cropping center
                    curr_center_v, curr_center_u = img.shape[0]//2, img.shape[1]//2
                    curr_crop_center = [curr_center_v, curr_center_u]

                    
                    img_cropped, dpt_cropped, bbox = self.crop(img, dpt, curr_crop_center)
                    K_adjusted = self._adjust_intrinsic(K, bbox)

                    
                    images.append(img_cropped)
                    depths.append(dpt_cropped)
                    intrinsics.append(K_adjusted)
                    poses.append(pose)
                    frame_ids.append(fid)
                    bboxes.append(bbox)
                    processed_images.append(self.transforms(img_cropped))
        elif self.crop_or_scale == 'scale':
            # All frames use scale processing
            for i in range(self.seq_len):
                img, dpt, K, pose, fid = original_data[i]
                img_scaled, dpt_scaled, K_scaled = self.scale(img, dpt, K)
                images.append(img_scaled)
                depths.append(dpt_scaled)
                intrinsics.append(K_scaled)
                poses.append(pose)
                frame_ids.append(fid)
                bboxes.append(np.array([0., 0.]))
                processed_images.append(self.transforms(img_scaled))
        elif self.crop_or_scale == 'scale_rotate':
            for i in range(self.seq_len):
                img, dpt, K, pose, fid = original_data[i]
                img_scaled, dpt_scaled, K_scaled = self.scale(img, dpt, K)
                images.append(img_scaled)
                depths.append(dpt_scaled)
                intrinsics.append(K_scaled)
                poses.append(pose)
                frame_ids.append(fid)
                bboxes.append(np.array([0., 0.]))  # scale without crop, offset is 0
                processed_images.append(self.transforms(img_scaled))

            # 2.1 Probabilistic sampling of rotation angle (fixed seed ensures reproducibility)
            np.random.seed(self.seed + idx)  # Each sample uses different seed to avoid global random conflicts
            rotate_angle = np.random.choice(self.rotate_angles, p=self.rotate_probs)
            
            # 2.2 Extract scaled reference frame data
            img_ref = images[ref_idx].copy()       # Scaled reference frame image
            dpt_ref = depths[ref_idx].copy()       # Scaled reference frame depth map
            K_ref = intrinsics[ref_idx].copy()     # Scaled reference frame intrinsic parameters
            S = self.image_size                    # Image size (S x S)

            # 2.3 Rotate reference frame image and depth map (ensure pixel correspondence)
            if rotate_angle != 0:
                # Image rotation (cv2.rotate maintains size, pixel correspondence)
                img_ref_rotated = cv2.rotate(img_ref, self.angle2cv_rotate[rotate_angle])
                # Depth map rotation (exactly same as image rotation, ensure spatial correspondence)
                dpt_ref_rotated = cv2.rotate(dpt_ref, self.angle2cv_rotate[rotate_angle])
            else:
                # No rotation: directly reuse original data
                img_ref_rotated = img_ref
                dpt_ref_rotated = dpt_ref

            # 2.4 Adjust reference frame intrinsic parameters (adapt to rotated image coordinates)
            K_ref_rotated = self._adjust_intrinsic_for_rotation(
                K_ref, rotate_angle, img_size=(S, S)
            )

            # 2.5 Update reference frame data in storage list (replace original scaled data)
            images[ref_idx] = img_ref_rotated
            depths[ref_idx] = dpt_ref_rotated
            intrinsics[ref_idx] = K_ref_rotated
            processed_images[ref_idx] = self.transforms(img_ref_rotated)  # Convert to Tensor again
        
        if self.crop_or_scale == 'scale_rotate':
            # 2.4 Adjust reference frame intrinsic parameters (original code, keep)
            K_ref_rotated = self._adjust_intrinsic_for_rotation(K_ref, rotate_angle, img_size=(S, S))

            # New: 2.4.1 Calculate rotation matrix, and extend to 4x4 pose matrix (homogeneous coordinates)
            R_rot = self._get_rotation_matrix(rotate_angle)
            T_rot = np.eye(4)
            T_rot[:3, :3] = R_rot  # Embed rotation matrix into pose matrix (translation is 0, only rotation)

            # New: 2.4.2 Adjust reference frame absolute pose (poses[ref_idx])
            # Assume poses is "camera→world" transformation (T_cw), then new pose = rotation matrix @ original pose
            poses[ref_idx] = T_rot @ poses[ref_idx]

            # 2.5 Update reference frame data in storage list (original code, keep)
            images[ref_idx] = img_ref_rotated
            depths[ref_idx] = dpt_ref_rotated
            intrinsics[ref_idx] = K_ref_rotated
            processed_images[ref_idx] = self.transforms(img_ref_rotated)

            # New: 2.6 Recalculate relative poses (rel_poses_to_ref), cancel reference frame rotation effect
            rel_poses_to_ref = []
            for i in range(1, self.seq_len):
                # New relative pose = original relative pose @ rotation matrix transpose (R_rot.T = R_rot inverse, because rotation matrix is orthogonal)
                rel_pose = poses[i] @ np.linalg.inv(poses[ref_idx])
                rel_poses_to_ref.append(rel_pose)
        else:
            # Calculate relative pose between reference frame and target frame
            rel_poses_to_ref = []
            for i in range(1, self.seq_len):
                rel_pose = poses[i] @ np.linalg.inv(poses[ref_idx])
                rel_poses_to_ref.append(rel_pose)

        ret = {
            'images': torch.stack(processed_images),  # [5, 3, H, W]
            'frame_ids': frame_ids,
            'overlaps': torch.from_numpy(np.array(scores, dtype=np.float32)),  # [4]
            'bboxes': torch.from_numpy(np.stack(bboxes).astype(np.float32)),  # [5, 2]
            'intrinsics': torch.from_numpy(np.stack(intrinsics).astype(np.float32)),  # [5, 3, 3]
            'abs_poses': torch.from_numpy(np.stack(poses).astype(np.float32)),  # [5, 4, 4]
            'rel_poses_to_ref': torch.from_numpy(np.stack(rel_poses_to_ref).astype(np.float32)),  # [4, 4, 4]
            'depths': torch.from_numpy(np.stack(depths).astype(np.float32)),  # [5, H, W]
            'ref_idx': ref_idx  # Reference frame index fixed to 0
        }

        
        return ret



import cv2
import numpy as np
import os

def save_projection_images(ret, sample_idx=0, out_dir='./vis_results', num_points=100, point_size=5):
    os.makedirs(out_dir, exist_ok=True)
    images = ret['images']  # [5, 3, H, W]
    depths = ret['depths']  # [5, H, W]
    intrinsics = ret['intrinsics']  # [5, 3, 3]
    rel_poses_to_ref = ret['rel_poses_to_ref']  # [4, 4, 4]
    ref_idx = ret['ref_idx']

    img_ref = images[ref_idx].permute(1,2,0).cpu().numpy()  # [H, W, 3]
    depth_ref = depths[ref_idx].cpu().numpy()
    intr_ref = intrinsics[ref_idx].cpu().numpy()
    H, W = depth_ref.shape

    
    np.random.seed(42)
    pts_2d = np.stack([
        np.random.randint(0, W, size=(num_points,)),
        np.random.randint(0, H, size=(num_points,))
    ], axis=1)  # [num_points, 2]

   
    img_ref_vis = (img_ref * 255).astype(np.uint8).copy()
    for x, y in pts_2d:
        cv2.circle(img_ref_vis, (int(x), int(y)), point_size, (0,255,0), -1)
    cv2.imwrite(os.path.join(out_dir, f'sample_{sample_idx}_ref.png'), cv2.cvtColor(img_ref_vis, cv2.COLOR_RGB2BGR))

    
    for idx_tgt in range(1, len(images)):
        img_tgt = images[idx_tgt].permute(1,2,0).cpu().numpy()
        intr_tgt = intrinsics[idx_tgt].cpu().numpy()
        pose_ref2tgt = rel_poses_to_ref[idx_tgt-1].cpu().numpy()

       
        pts_homo = np.hstack([pts_2d, np.ones((num_points, 1))]).T  # [3, N]
        depth_sampled = depth_ref[pts_2d[:, 1], pts_2d[:, 0]]
        pts_cam = np.linalg.inv(intr_ref) @ pts_homo * depth_sampled
        pts_cam = np.vstack([pts_cam, np.ones((1, num_points))])  # [4, N]
        pts_cam_tgt = pose_ref2tgt @ pts_cam
        pts_cam_tgt = pts_cam_tgt[:3]
        pts_proj = intr_tgt @ pts_cam_tgt
        pts_proj = pts_proj[:2] / pts_proj[2:]

        img_tgt_vis = (img_tgt * 255).astype(np.uint8).copy()
        valid = (pts_proj[0] >= 0) & (pts_proj[0] < W) & (pts_proj[1] >= 0) & (pts_proj[1] < H)
        for x, y, v in zip(pts_proj[0], pts_proj[1], valid):
            if v:
                cv2.circle(img_tgt_vis, (int(x), int(y)), point_size, (255,0,0), -1)
        cv2.imwrite(os.path.join(out_dir, f'sample_{sample_idx}_tgt{idx_tgt}.png'), cv2.cvtColor(img_tgt_vis, cv2.COLOR_RGB2BGR))