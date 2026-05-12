# based on: https://github.com/uoip/monoVO-python

import numpy as np
import cv2
import logging
import glob
import torch
import torch.nn as nn
# First import DualSoftmaxMatcher (keep original implementation unchanged)
class DualSoftmaxMatcher(nn.Module):
    def __init__(self, inv_temperature=20, thr=0.01):
        super().__init__()
        self.inv_temperature = inv_temperature
        self.thr = thr

    def forward(self, info0, info1, thr=None):
        desc0 = info0['descriptors']
        desc1 = info1['descriptors']
        
        inds, P = self.dual_softmax(desc0, desc1, thr)
        mkpts0 = info0['keypoints'][inds[:, 0]]
        mkpts1 = info1['keypoints'][inds[:, 1]]
        mconf = P[inds[:, 0], inds[:, 1]]
        
        return mkpts0, mkpts1, mconf, inds  # Add return inds for convenient subsequent index construction

    def dual_softmax(self, desc0, desc1, thr=None):
        if thr is None:
            thr = self.thr
        dist_mat = (desc0 @ desc1.t()) * self.inv_temperature
        P = dist_mat.softmax(dim=-2) * dist_mat.softmax(dim=-1)
        
        # Bidirectional maximum matching + threshold filtering
        inds = torch.nonzero(
            (P == P.max(dim=-1, keepdim=True).values) &
            (P == P.max(dim=-2, keepdim=True).values) &
            (P >= thr)
        )
        return inds, P

def create_dataloader(conf):
    try:
        code_line = f"{conf['name']}(conf)"
        loader = eval(code_line)
    except NameError:
        raise NotImplementedError(f"{conf['name']} is not implemented yet.")

    return loader

"""
Pinhole camera model class: used to define pinhole camera intrinsic parameters
fx,fy: focal length
cx,cy: principal point position
k1,k2,p1,p2,p3: distortion parameters
"""
class PinholeCamera(object):
    def __init__(self, width, height, fx, fy, cx, cy,
                 k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]

class KITTILoader(object):
    default_config = {
        "root_path": "../test_imgs",
        "sequence": "00",
        "start": 0
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("KITTI Dataset config: ")
        logging.info(self.config)

        if self.config["sequence"] in ["00", "01", "02"]:
            self.cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
        elif self.config["sequence"] in ["03"]:
            self.cam = PinholeCamera(1242.0, 375.0, 721.5377, 721.5377, 609.5593, 172.854)
        elif self.config["sequence"] in ["04", "05", "06", "07", "08", "09", "10"]:
            self.cam = PinholeCamera(1226.0, 370.0, 707.0912, 707.0912, 601.8873, 183.1104)
        else:
            raise ValueError(f"Unknown sequence number: {self.config['sequence']}")

        # read ground truth pose
        self.pose_path = self.config["root_path"] + "/poses/" + self.config["sequence"] + ".txt"
        self.gt_poses = []
        with open(self.pose_path) as f:
            lines = f.readlines()
            for line in lines:
                ss = line.strip().split()
                pose = np.zeros((1, len(ss)))
                for i in range(len(ss)):
                    pose[0, i] = float(ss[i])

                pose.resize([3, 4])
                self.gt_poses.append(pose)
        
        # image id
        self.img_id = self.config["start"]
        self.img_N = len(glob.glob(pathname=self.config["root_path"] + "/sequences/" \
                                            + self.config["sequence"] + "/image_0/*.png"))
        
    def get_cur_pose(self):
        return self.gt_poses[self.img_id - 1]

    def __getitem__(self, item):
        file_name = self.config["root_path"] + "/sequences/" + self.config["sequence"] \
                    + "/image_0/" + str(item).zfill(6) + ".png"
        img = cv2.imread(file_name)
        return img

    def __iter__(self):
        return self

    def __next__(self):
        if self.img_id < self.img_N:
            file_name = self.config["root_path"] + "/sequences/" + self.config["sequence"] \
                        + "/image_0/" + str(self.img_id).zfill(6) + ".png"
            img = cv2.imread(file_name)

            self.img_id += 1

            return img
        raise StopIteration()

    def __len__(self):
        return self.img_N - self.config["start"]
    

def create_detector(conf):
    try:
        code_line = f"{conf['name']}(conf)"
        detector = eval(code_line)
    except NameError:
        raise NotImplementedError(f"{conf['name']} is not implemented yet.")

    return detector


def create_matcher(conf):
    try:
        code_line = f"{conf['name']}(conf)"
        matcher = eval(code_line)
    except NameError:
        raise NotImplementedError(f"{conf['name']} is not implemented yet.")

    return matcher



# Modified FrameByFrameMatcher
class FrameByFrameMatcher(object):
    default_config = {
        "type": "FLANN",
        "KNN": {
            "HAMMING": True,
            "first_N": 300,
        },
        "FLANN": {
            "kdTrees": 5,
            "searchChecks": 50
        },
        "DualSoftmax": {  # Add DualSoftmax configuration
            "inv_temperature": 20,
            "thr": 0.01,
            "device": "cpu"  # Support CPU/GPU (requires PyTorch environment)
        },
        "distance_ratio": 0.75
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("Frame by frame matcher config: ")
        logging.info(self.config)
        self.matcher_type = self.config["type"].lower()  # Unify lowercase to avoid case errors
        self.mconf = None  # Store DualSoftmax confidence

        if self.matcher_type == "knn":
            logging.info("creating brutal force matcher...")
            if self.config["KNN"]["HAMMING"]:
                logging.info("brutal force with hamming norm.")
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            else:
                self.matcher = cv2.BFMatcher()
        elif self.matcher_type == "flann":
            logging.info("creating FLANN matcher...")
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=self.config["FLANN"]["kdTrees"])
            search_params = dict(checks=self.config["FLANN"]["searchChecks"])
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif self.matcher_type == "dualsoftmaxmatch":  # Add DualSoftmax matcher initialization
            logging.info("creating DualSoftmax matcher...")
            # Initialize DualSoftmaxMatcher, specify device
            self.device = torch.device(self.config["DualSoftmax"]["device"])
            self.matcher = DualSoftmaxMatcher(
                inv_temperature=self.config["DualSoftmax"]["inv_temperature"],
                thr=self.config["DualSoftmax"]["thr"]
            ).to(self.device)
        else:
            raise ValueError(f"Unknown matcher type: {self.matcher_type}")

    def match(self, kptdescs):
        self.good = []
        # Handle DualSoftmax matching (separate branch, adapt to PyTorch)
        if self.matcher_type == "dualsoftmaxmatch":
            logging.debug("DualSoftmax keypoints matching...")
            # 1. Extract ref/cur features (NumPy arrays to PyTorch tensors)
            info0 = {
                "descriptors": torch.from_numpy(kptdescs["ref"]["descriptors"]).float().to(self.device),
                "keypoints": torch.from_numpy(kptdescs["ref"]["keypoints"]).float().to(self.device)
            }
            info1 = {
                "descriptors": torch.from_numpy(kptdescs["cur"]["descriptors"]).float().to(self.device),
                "keypoints": torch.from_numpy(kptdescs["cur"]["keypoints"]).float().to(self.device)
            }
            # 2. Execute DualSoftmax matching (reuse original forward method, add return inds)
            mkpts0, mkpts1, self.mconf, inds = self.matcher.forward(info0, info1)
            # 3. Construct self.good consistent with original format (compatible with get_good_keypoints)
            # Simulate cv2.DMatch object: only keep queryIdx (ref index) and trainIdx (cur index)
            for ref_idx, cur_idx in inds.cpu().numpy():  # Convert to NumPy for subsequent processing
                # Use simple class to simulate DMatch, avoid dependency on cv2 objects
                class MockDMatch:
                    def __init__(self, queryIdx, trainIdx):
                        self.queryIdx = queryIdx
                        self.trainIdx = trainIdx
                self.good.append([MockDMatch(ref_idx, cur_idx)])
            return self.good

        # Original KNN/FLANN matching logic (keep unchanged)
        self.descriptor_shape = kptdescs["ref"]["descriptors"].shape[1]
        if self.matcher_type == "knn" and self.config["KNN"]["HAMMING"]:
            logging.debug("KNN keypoints matching...")
            matches = self.matcher.match(kptdescs["ref"]["descriptors"], kptdescs["cur"]["descriptors"])
            matches = sorted(matches, key=lambda x: x.distance)
            # Fix boundary error: avoid index out of bounds when matches are fewer than first_N
            take_n = min(self.config["KNN"]["first_N"], len(matches))
            for i in range(take_n):
                self.good.append([matches[i]])
        else:
            logging.debug("FLANN keypoints matching...")
            matches = self.matcher.knnMatch(kptdescs["ref"]["descriptors"], kptdescs["cur"]["descriptors"], k=2)
            for m, n in matches:
                if m.distance < self.config["distance_ratio"] * n.distance:
                    self.good.append([m])
            self.good = sorted(self.good, key=lambda x: x[0].distance)
        return self.good

    def get_good_keypoints(self, kptdescs):
        logging.debug("getting matched keypoints...")
        kp_ref = np.zeros([len(self.good), 2])
        kp_cur = np.zeros([len(self.good), 2])
        match_dist = np.zeros([len(self.good)])  # Not used in DualSoftmax, only for format compatibility
        for i, m in enumerate(self.good):
            kp_ref[i, :] = kptdescs["ref"]["keypoints"][m[0].queryIdx]
            kp_cur[i, :] = kptdescs["cur"]["keypoints"][m[0].trainIdx]
            # DualSoftmax doesn't need to store dist, other types keep original logic
            if self.matcher_type != "dualsoftmaxmatch":
                match_dist[i] = m[0].distance

        ret_dict = {
            "ref_keypoints": kp_ref,
            "cur_keypoints": kp_cur,
            "match_score": self.normalised_matching_scores(match_dist)
        }
        return ret_dict

    def __call__(self, kptdescs):
        self.match(kptdescs)
        return self.get_good_keypoints(kptdescs)

    def normalised_matching_scores(self, match_dist):
        # DualSoftmax: directly return confidence (normalized, 0-1, higher is better)
        if self.matcher_type == "dualsoftmaxmatch":
            return self.mconf.cpu().numpy()  # Convert to NumPy consistent with original format

        # Original score calculation logic (keep unchanged)
        if self.matcher_type == "knn" and self.config["KNN"]["HAMMING"]:
            best, worst = 0, self.descriptor_shape * 8
            worst = worst / 4
        else:
            if match_dist.max() > 1:
                best, worst = 0, self.descriptor_shape * 2
            else:
                best, worst = 0, 1

        match_scores = match_dist / worst
        match_scores[match_scores > 1] = 1
        match_scores[match_scores < 0] = 0
        match_scores = 1 - match_scores

        return match_scores

    def draw_matched(self, img0, img1):
        pass

# --- VISUALIZATION ---
# based on: https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/utils.py
def plot_keypoints(image, kpts):
    kpts = np.round(kpts).astype(int)
    for x, y in kpts:
        cv2.drawMarker(image, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 6)

    return image

class VisualOdometry(object):
    def __init__(self, detector, matcher, cam):
        self.detector = detector
        self.matcher = matcher
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.index = 0
        self.kptdescs = {}
        self.match_kps = {}
        self.cur_R = None
        self.cur_t = None

    def update(self, image, absolute_scale=1):
        predict_data = self.detector.extract_vo_v2(image)
        
        kptdesc = {
            "keypoints": predict_data["keypoints"].cpu().detach().numpy(),
            "descriptors": predict_data["descriptors"].cpu().detach().numpy()
        }
        if self.index == 0:
            self.kptdescs["cur"] = kptdesc
            self.cur_R = np.identity(3)
            self.cur_t = np.zeros((3, 1))
        else:
            self.kptdescs["cur"] = kptdesc
            matches = self.matcher(self.kptdescs)
            self.match_kps = {"cur":matches['cur_keypoints'], "ref":matches['ref_keypoints']}
            E, mask = cv2.findEssentialMat(
                matches['cur_keypoints'],
                matches['ref_keypoints'],
                focal=self.focal,
                pp=self.pp,
                method=cv2.RHO,
                prob=0.999,
                threshold=1.0
            )
            _, R, t, mask = cv2.recoverPose(E, matches['cur_keypoints'], matches['ref_keypoints'],
                                            focal=self.focal, pp=self.pp)
            t[1] = -t[1]
            if (absolute_scale > 0.1):
                self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
                self.cur_R = self.cur_R.dot(R)
        self.kptdescs["ref"] = self.kptdescs["cur"]
        self.index += 1
        return self.cur_R, self.cur_t

class AbosluteScaleComputer(object):
    def __init__(self):
        self.prev_pose = None
        self.cur_pose = None
        self.count = 0

    def update(self, pose):
        self.cur_pose = pose
        scale = 1.0
        if self.count != 0:
            scale = np.sqrt(
                (self.cur_pose[0, 3] - self.prev_pose[0, 3]) * (self.cur_pose[0, 3] - self.prev_pose[0, 3])
                + (self.cur_pose[1, 3] - self.prev_pose[1, 3]) * (self.cur_pose[1, 3] - self.prev_pose[1, 3])
                + (self.cur_pose[2, 3] - self.prev_pose[2, 3]) * (self.cur_pose[2, 3] - self.prev_pose[2, 3]))
        self.count += 1
        self.prev_pose = self.cur_pose
        return scale



