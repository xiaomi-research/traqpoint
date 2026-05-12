# Copyright (C) 2026 Xiaomi Corporation.
# Copyright (C) 2024 ETH Zurich (Hierarchical-Localization)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import pprint
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Dict, List, Optional, Tuple, Union
from torch import nn
import h5py
import torch
from tqdm import tqdm

from hloc import logger
from hloc.utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval



class DualSoftmaxMatcher(nn.Module):
    def __init__(self, inv_temperature = 20, thr = 0.01):
        super().__init__()
        self.inv_temperature = inv_temperature
        self.thr = thr

    def forward(self, info0, info1, thr = None):
        desc0 = info0['descriptors']
        desc1 = info1['descriptors']
        
        inds, P = self.dual_softmax(desc0, desc1, thr)

         # Initialize match results (shape [N], N is the number of desc0 features)
        N = desc0.shape[0]
        matches0 = torch.full((1, N), -1, dtype=torch.long, device=desc0.device)
        matching_scores0 = torch.zeros((1, N), device=desc0.device)

        # Fill match results (batch dimension fixed at 0)
        if inds.numel() > 0:
            matches0[0, inds[:, 0]] = inds[:, 1]  # batch index fixed at 0
            matching_scores0[0, inds[:, 0]] = P[inds[:, 0], inds[:, 1]]

        # Return output in specified format
        return {
            "matches0": matches0,
            "matching_scores0": matching_scores0
        }

    def dual_softmax(self, desc0, desc1, thr = None):
        if thr is None:
            thr = self.thr
        dist_mat = (desc0 @ desc1.t()) * self.inv_temperature
        P = dist_mat.softmax(dim = -2) * dist_mat.softmax(dim= -1)
        
        inds = torch.nonzero((P == P.max(dim=-1, keepdim = True).values) 
                        * (P == P.max(dim=-2, keepdim = True).values) * (P >= thr))
        
        return inds, P



class Matcher(nn.Module):
    default_conf = {
        "inv_temperature": 20,  # DualSoftmax parameter
        "thr": 0.01,            # matching threshold
    }
    
    required_inputs = [
        "keypoints0",
        "descriptors0",
        "keypoints1",
        "descriptors1",
    ]
    
    def __init__(self, conf):
        super().__init__()
        # Initialize DualSoftmaxMatcher
        self.net = DualSoftmaxMatcher(
            inv_temperature=conf.get("inv_temperature", 20),
            thr=conf.get("thr", 0.01)
        )
    
    def forward(self, data):
        """Check input data and invoke matching logic"""
        for key in self.required_inputs:
            assert key in data, f"Missing key {key} in data"
        return self._forward(data)
    
    def _forward(self, data):
        # Format input data (adapted for DualSoftmaxMatcher input requirements)
        info0 = {
            "descriptors": data["descriptors0"].squeeze(0),
            "keypoints": data["keypoints0"].squeeze(0)
        }
        info1 = {
            "descriptors": data["descriptors1"].squeeze(0),
            "keypoints": data["keypoints1"].squeeze(0)
        }

        if info0["descriptors"].shape[0] == 0 or info1["descriptors"].shape[0] == 0:
            N = info0["descriptors"].shape[0]
            # Return an empty but correctly formatted dict
            return {
                "matches0": torch.full((1, N), -1, dtype=torch.long, device=data["descriptors0"].device),
                "matching_scores0": torch.zeros((1, N), device=data["descriptors0"].device)
            }
        
        # Invoke DualSoftmaxMatcher for matching
        
        pred = self.net(info0, info1)
        
        return pred
        
        # # Build match results (adapted for original code output format)
        # # Generate matches0: each element is the matched index in image1
        # # First build keypoint index mapping
        # kpts0 = data["keypoints0"][0].cpu().numpy()
        # kpts1 = data["keypoints1"][0].cpu().numpy()
        # matches0 = []
        # for pt0 in mkpts0.cpu().numpy():
        #     # Find the index in original keypoints
        #     idx0 = (kpts0 == pt0).all(axis=1).nonzero()[0][0]
        #     matches0.append(idx0)
        # for pt1 in mkpts1.cpu().numpy():
        #     idx1 = (kpts1 == pt1).all(axis=1).nonzero()[0][0]
        #     matches0.append(idx1)
        
        # # Convert to the output format expected by original code
        # return {
        #     "matches0": torch.tensor(matches0, device=data["keypoints0"].device).reshape(1, -1, 2).long(),
        #     "matching_scores0": conf.reshape(1, -1)  # confidence scores
        # }
        

"""
Updated configuration, removed LightGlue-related parameters, using DualSoftmax config
"""
confs = {
    "traqpoint+dual_softmax": {
        "output": "matches-traqpoint-dual_softmax",
        "model": {
            "name": "dual_softmax",
            "inv_temperature": 20,
            "thr": 0.01,
        },
    }
}


class WorkQueue:
    def __init__(self, work_fn, num_threads=1):
        self.queue = Queue(num_threads)
        self.threads = [
            Thread(target=self.thread_fn, args=(work_fn,)) for _ in range(num_threads)
        ]
        for thread in self.threads:
            thread.start()

    def join(self):
        for thread in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join()

    def thread_fn(self, work_fn):
        item = self.queue.get()
        while item is not None:
            work_fn(item)
            item = self.queue.get()

    def put(self, data):
        self.queue.put(data)


class FeaturePairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, feature_path_q, feature_path_r):
        self.pairs = pairs
        self.feature_path_q = feature_path_q
        self.feature_path_r = feature_path_r

    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]
        data = {}
        with h5py.File(self.feature_path_q, "r") as fd:
            grp = fd[name0]
            for k, v in grp.items():
                data[k + "0"] = torch.from_numpy(v.__array__()).float()
        with h5py.File(self.feature_path_r, "r") as fd:
            grp = fd[name1]
            for k, v in grp.items():
                data[k + "1"] = torch.from_numpy(v.__array__()).float()
        return data

    def __len__(self):
        return len(self.pairs)


def writer_fn(inp, match_path):
    pair, pred = inp
    with h5py.File(str(match_path), "a", libver="latest") as fd:
        if pair in fd:
            del fd[pair]
        grp = fd.create_group(pair)
        
        matches = pred["matches0"][0].cpu().short().numpy()
        grp.create_dataset("matches0", data=matches)
        if "matching_scores0" in pred:
            scores = pred["matching_scores0"][0].cpu().half().numpy()
            grp.create_dataset("matching_scores0", data=scores)


def main(
    conf: Dict,
    pairs: Path,
    features: Union[Path, str],
    export_dir: Optional[Path] = None,
    matches: Optional[Path] = None,
    features_ref: Optional[Path] = None,
    overwrite: bool = False,
    device: str = "cpu",
) -> Path:
    if isinstance(features, Path) or Path(features).exists():
        features_q = features
        if matches is None:
            raise ValueError(
                "Either provide both features and matches as Path" " or both as names."
            )
    else:
        if export_dir is None:
            raise ValueError(
                "Provide an export_dir if features is not" f" a file path: {features}."
            )
        features_q = Path(export_dir, features + ".h5")
        if matches is None:
            matches = Path(export_dir, f'{features}_{conf["output"]}_{pairs.stem}.h5')

    if features_ref is None:
        features_ref = features_q
    match_from_paths(conf, pairs, matches, features_q, features_ref, overwrite)

    return matches


def find_unique_new_pairs(pairs_all: List[Tuple[str]], match_path: Path = None):
    """Avoid redundant computation"""
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = list(pairs)
    if match_path is not None and match_path.exists():
        with h5py.File(str(match_path), "r", libver="latest") as fd:
            pairs_filtered = []
            for i, j in pairs:
                if (
                    names_to_pair(i, j) in fd
                    or names_to_pair(j, i) in fd
                    or names_to_pair_old(i, j) in fd
                    or names_to_pair_old(j, i) in fd
                ):
                    continue
                pairs_filtered.append((i, j))
        return pairs_filtered
    return pairs


@torch.no_grad()
def match_from_paths(
    conf: Dict,
    pairs_path: Path,
    match_path: Path,
    feature_path_q: Path,
    feature_path_ref: Path,
    overwrite: bool = False,
) -> Path:
    logger.info(
        "Matching local features with configuration:" f"\n{pprint.pformat(conf)}"
    )   
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if not feature_path_q.exists():
        raise FileNotFoundError(f"Query feature file {feature_path_q}.")
    if not feature_path_ref.exists():
        raise FileNotFoundError(f"Reference feature file {feature_path_ref}.")
    match_path.parent.mkdir(exist_ok=True, parents=True)

    assert pairs_path.exists(), pairs_path
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    if len(pairs) == 0:
        logger.info("Skipping the matching.")
        return

    # Initialize DualSoftmax matcher
    model = Matcher(conf["model"])
    model.eval()
    model.to(device)

    dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=5, batch_size=1, shuffle=False, pin_memory=True
    )
    writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 5)

    for idx, data in enumerate(tqdm(loader, smoothing=0.1)):
        # Move data to device
        data = {
            k: v.to(device, non_blocking=True)
            for k, v in data.items()
        }
        pred = model(data)
        
        # # Filter pairs with too few matches
        # if pred["matches0"].shape[1] < 25:
        #     continue
            
        pair = names_to_pair(*pairs[idx])
        writer_queue.put((pair, pred))
    writer_queue.join()
    logger.info("Finished exporting matches.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--export_dir", type=Path)
    parser.add_argument("--features", type=str, default="feats-traqpoint-n4096")
    parser.add_argument("--matches", type=Path)
    parser.add_argument(
        "--conf", type=str, default="traqpoint+dual_softmax", choices=list(confs.keys())
    )  # update default config
    args = parser.parse_args()
    main(confs[args.conf], args.pairs, args.features, args.export_dir)