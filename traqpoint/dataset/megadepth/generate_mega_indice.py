import os
import numpy as np
import random
from glob import glob
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def generate_sequence_npz_megadepth(
    npz_path,           # Path to the npz file for a single scene
    save_dir,           # Directory to save the generated sequence npz files
    window_size=150,    # Maximum time interval (index difference) between auxiliary ID and head ID
    min_score=0.1,      # Minimum overlap score threshold
    max_score=0.7,      # Maximum overlap score threshold
    seq_len=5           # Sequence length
):
    scene_name = os.path.splitext(os.path.basename(npz_path))[0]
    try:
        # 1. Load original data
        scene_info = dict(np.load(npz_path, allow_pickle=True))
        pair_infos = scene_info['pair_infos'].copy()  # [(idx0, idx1), score, ...]
        # Keep all meta-information except for pair_infos
        scene_info_to_save = {k: v for k, v in scene_info.items() if k != 'pair_infos'}

        # 2. Build a map from index to valid pairs
        id_pairs = defaultdict(list)
        for pair_info in pair_infos:
            (idx0, idx1), score, _ = pair_info
            if score < min_score or score > max_score:
                continue
            id_pairs[idx0].append((idx1, score))
            id_pairs[idx1].append((idx0, score))

        # 3. Filter for qualified head IDs (each head must have at least seq_len - 1 valid pairs)
        qualified_heads = []
        for head_id in id_pairs:
            valid_pairs = [(pid, score) for pid, score in id_pairs[head_id]
                           if abs(pid - head_id) <= window_size]
            if len(valid_pairs) >= seq_len - 1:
                qualified_heads.append((head_id, valid_pairs))

        if not qualified_heads:
            raise ValueError(f"No qualified head IDs found (requires at least {seq_len - 1} valid pairs)")

        # 4. Sample auxiliary IDs for each qualified head ID
        sequences = []
        sequence_scores = []
        for head_id, valid_pairs in qualified_heads:
            selected = random.sample(valid_pairs, k=seq_len - 1)
            aux_ids = [pid for pid, _ in selected]
            scores = [score for _, score in selected]
            sequences.append([head_id] + aux_ids)
            sequence_scores.append(scores)

        # 5. Save the results
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{scene_name}.npz")
        np.savez(
            save_path,
            sequence_infos=np.array(sequences, dtype=np.int32),      # Image indices for the sequence
            sequence_scores=np.array(sequence_scores, dtype=np.float32), # Corresponding overlap scores
            **scene_info_to_save
        )
        return scene_name, True, f"Generated {len(sequences)} sequences"

    except Exception as e:
        return scene_name, False, str(e)

def main():
    # Configuration parameters
    MEGADEPTH_ROOT = '/high_perf_store3/world-model/liuyepeng/data/open_source/megadepth' # Your MegaDepth root directory
    TRAIN_BASE_PATH = f"{MEGADEPTH_ROOT}/megadepth_indices"
    TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"
    SAVE_DIR = f"{TRAIN_BASE_PATH}/sequence_indices_0.1_0.7_s5" # Directory to save the sequence npz files
    window_size = 2000
    seq_len = 5
    min_score = 0.1
    max_score = 0.7
    max_workers = 16

    npz_paths = glob(TRAIN_NPZ_ROOT + '/*.npz')
    if not npz_paths:
        print("No scene npz files to process, exiting.")
        return

    print(f"Found {len(npz_paths)} scenes, starting multi-threaded processing...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                generate_sequence_npz_megadepth,
                npz_path=npz_path,
                save_dir=SAVE_DIR,
                window_size=window_size,
                min_score=min_score,
                max_score=max_score,
                seq_len=seq_len
            ): npz_path for npz_path in npz_paths
        }
        completed = 0
        success_count = 0
        for future in as_completed(futures):
            npz_path = futures[future]
            scene_name = os.path.splitext(os.path.basename(npz_path))[0]
            try:
                scene_name, success, message = future.result()
                completed += 1
                if success:
                    success_count += 1
                    print(f"[{completed}/{len(npz_paths)}] Success: {scene_name} - {message}")
                else:
                    print(f"[{completed}/{len(npz_paths)}] Failed: {scene_name} - {message}")
            except Exception as e:
                completed += 1
                print(f"[{completed}/{len(npz_paths)}] Error: {scene_name} - An unexpected error occurred while fetching the result: {str(e)}")

    end_time = time.time()
    print(f"\nProcessing complete! Total time: {end_time - start_time:.2f} seconds")
    print(f"Processing statistics: {success_count} successful, {len(npz_paths) - success_count} failed, {len(npz_paths)} total")

if __name__ == "__main__":
    main()