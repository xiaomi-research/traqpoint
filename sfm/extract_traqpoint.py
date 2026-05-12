# Copyright (C) 2026 Xiaomi Corporation.
# Copyright (C) 2024 ETH Zurich (Hierarchical-Localization)
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
import collections.abc as collections
import glob
import pprint
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Union
import cv2
import h5py
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
from hloc.extract_features import ImageDataset
from hloc import logger
from hloc.utils.base_model import dynamic_load
from hloc.utils.io import list_h5_names, read_image
from hloc.utils.parsers import parse_image_lists
from traqpoint.traqpoint_sfm import build
from traqpoint.utils import read_config

confs = {
    'traqpoint': {
        "output": "feats-TAP",
        "model": {
            'config_path': './configs/default.yaml',
            'weights': './weights/traqpoint_best.pth',
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1280,
            "resize_force": True,
        }
    }
}

@torch.no_grad()
def main(
    conf: Dict,
    image_dir: Path,
    export_dir: Optional[Path] = None,
    as_half: bool = True,
    image_list: Optional[Union[Path, List[str]]] = None,
    feature_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    logger.info(
        "Extracting local features with configuration:" f"\n{pprint.pformat(conf)}"
    )

    dataset = ImageDataset(image_dir, conf["preprocessing"], image_list)
    if feature_path is None:
        feature_path = Path(export_dir, conf["output"] + ".h5")
    feature_path.parent.mkdir(exist_ok=True, parents=True)
    skip_names = set(
        list_h5_names(feature_path) if feature_path.exists() and not overwrite else ()
    )
    dataset.names = [n for n in dataset.names if n not in skip_names]
    if len(dataset.names) == 0:
        logger.info("Skipping the extraction.")
        return feature_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = read_config(conf["model"]["config_path"])
    config['device'] = device
    model = build(config, conf["model"]["weights"])
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=1, shuffle=False, pin_memory=True
    )
    for idx, data in enumerate(tqdm(loader)):
        name = dataset.names[idx]
        features = model.extract(data["image"])
        
        pred = {
            "keypoints": [f["keypoints"] for f in features],
            "keypoint_scores": [f["scores"] for f in features],
            "descriptors": [f["descriptors"] for f in features],
        }
        
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        pred["image_size"] = original_size = data["original_size"][0].numpy()
        if "keypoints" in pred:
            size = np.array(data["image"].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            pred["keypoints"] = (pred["keypoints"] + 0.5) * scales[None] - 0.5
            if "scales" in pred:
                pred["scales"] *= scales.mean()
            # add keypoint uncertainties scaled to the original resolution
            uncertainty = getattr(model, "detection_noise", 1) * scales.mean()

            # --- filter: remove kpts within 5px from left/right borders (in original image coords) ---
            margin = 8
            w = int(original_size[0])  # original width
            if pred["keypoints"].size > 0:
                xs = pred["keypoints"][:, 0]
                mask = (xs >= margin) & (xs <= (w - 1 - margin))

                if mask.ndim == 1:
                    kept = int(mask.sum())
                    total = pred["keypoints"].shape[0]
                    if kept != total:
                        # keypoints
                        pred["keypoints"] = pred["keypoints"][mask]
                        # scores
                        if "keypoint_scores" in pred and pred["keypoint_scores"].ndim == 1 and pred["keypoint_scores"].shape[0] == total:
                            pred["keypoint_scores"] = pred["keypoint_scores"][mask]
                        # descriptors: support (K, D) or (D, K)
                        if "descriptors" in pred and pred["descriptors"].ndim == 2:
                            desc = pred["descriptors"]
                            if desc.shape[0] == total:
                                pred["descriptors"] = desc[mask]
                            elif desc.shape[1] == total:
                                pred["descriptors"] = desc[:, mask]
                        print(f"[extract_traqpoint] filtered border kpts: kept {kept}/{total} (margin={margin}px, width={w})")
                # ...existing code...

        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)

        with h5py.File(str(feature_path), "a", libver="latest") as fd:
            try:
                if name in fd:
                    del fd[name]
                grp = fd.create_group(name)
                for k, v in pred.items():
                    grp.create_dataset(k, data=v)
                if "keypoints" in pred:
                    grp["keypoints"].attrs["uncertainty"] = uncertainty
            except OSError as error:
                if "No space left on device" in error.args[0]:
                    logger.error(
                        "Out of disk space: storing features on disk can take "
                        "significant space, did you enable the as_half flag?"
                    )
                    del grp, fd[name]
                raise error

        del pred

    logger.info("Finished exporting features.")
    print("Output features saved to:", feature_path)
    return feature_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--export_dir", type=Path, required=True)
    parser.add_argument(
        "--conf", type=str, default="traqpoint", choices=list(confs.keys())
    )
    parser.add_argument("--as_half", action="store_true")
    parser.add_argument("--image_list", type=Path)
    parser.add_argument("--feature_path", type=Path)
    args = parser.parse_args()
    main(
        confs[args.conf],
        args.image_dir,
        args.export_dir,
        args.as_half,
        args.image_list,
        args.feature_path,
    )