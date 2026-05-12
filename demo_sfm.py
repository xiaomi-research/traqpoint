"""Minimal TraqPoint SfM bundle entrypoint.

Requires an external, importable `hloc` installation/environment.
Run this script from the `traqpoint_sfm_bundle/` directory so relative paths such as
`./configs/default.yaml` and `./outputs/...` resolve correctly.
"""

from sfm import match_traqpoint_ds, extract_traqpoint
from hloc import (
    extract_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
    pairs_from_exhaustive,
)
from pathlib import Path
import os
import torch
import sys

# Madrid_Metropolis Gendarmenmarkt Tower_of_London
images_dir = Path('/high_perf_store2/users/liuyepeng/code/eth-benchmarks/eth-benchmark/Madrid_Metropolis/images')
device = torch.cuda.is_available()
images = [image for image in os.listdir(images_dir) if image.endswith('.jpg') or image.endswith('.png')]
outputs = Path('./outputs/reconstruction/Madrid_Metropolis_16000_0_35_raw_2_filter_8_1280_smoketest')
if not outputs.exists():
    outputs.mkdir(parents=True)
sfm_pairs = outputs / 'sfm_pairs.txt'
retrieval_conf = extract_features.confs["netvlad"]
feature_conf = extract_traqpoint.confs["traqpoint"]
matcher_conf = match_traqpoint_ds.confs["traqpoint+dual_softmax"]
exhaustive_if_less = 30
num_matched = 25

# Check whether to skip earlier steps and go directly to reconstruction
skip_to_reconstruction = '--skip-to-reconstruction' in sys.argv

if not skip_to_reconstruction:
    # image_retrieval
    if len(images) < exhaustive_if_less:
        pairs_from_exhaustive.main(sfm_pairs, images)
    else:
        retrieval_path = extract_features.main(retrieval_conf, images_dir, outputs)
        pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=num_matched)

    # feature_extraction
    feature_path = extract_traqpoint.main(feature_conf, images_dir, outputs, overwrite=True)
    # feature_path = extract_traqpoint.main(feature_conf, images_dir, outputs)

    # matching
    match_path = match_traqpoint_ds.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)
else:
    # Use existing files directly
    feature_path = outputs / "feats-TAP.h5"
    # Adjust match file path based on actual filename
    match_path = outputs / "feats-TAP_matches-traqpoint-dual_softmax_sfm_pairs.h5"
    print(f"Skipping earlier steps, using existing files:")
    print(f"  Feature file: {feature_path}")
    print(f"  Match file: {match_path}")
    print(f"  Pairs file: {sfm_pairs}")


# # reconstruction
image_options = {}
mapper_options = {}
model = reconstruction.main(outputs, images_dir, sfm_pairs, feature_path, 
            match_path, verbose=True, camera_mode='PER_IMAGE', image_options=image_options, mapper_options=mapper_options,
            min_match_score = 0.2, skip_geometric_verification=False)

# print(model.summary())
# Print parameters used in this experiment
print("--16000_0_35_raw_2_filter_8_1280--")
visualization.visualize_sfm_2d(model, images_dir, color_by="depth", n=5)
