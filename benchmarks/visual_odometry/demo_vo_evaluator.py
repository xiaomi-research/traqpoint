import numpy as np
import cv2
import argparse
import os
import sys
import torch
import yaml

# Add traqpoint-official to Python path
sys.path.append(".")

from benchmarks.visual_odometry.track_utils.VisualOdometry import VisualOdometry, AbosluteScaleComputer, create_dataloader, \
    plot_keypoints, create_matcher

# Import from traqpoint-official
from traqpoint.traqpoint import TraqPoint
from traqpoint.models.detector import build_detector
from traqpoint.models.descriptor import build_descriptor

def build_vo_model(detection_threshold=0.5, weights_path=None):
    """Build TraqPoint model for Visual Odometry with local weights"""
    # Inline configuration from traqpoint-official/configs/default.yaml
    config = {
        'activation': 'relu',
        'block_dims': [8, 16, 32, 64],
        'd_model': 256,
        'detection_threshold': detection_threshold,
        'device': 'cuda',
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'enc_n_points': 8,
        'hidden_dim': 256,
        'lr_backbone': 0,
        'nhead': 8,
        'num_encoder_layers': 4,
        'num_feature_levels': 4,
        'top_k': 4096,
        'train_detector': False
    }

    if weights_path is None:
        weights_path = "weights/traqpoint_best.pth"

    print(f'Building VO model with detection_threshold={detection_threshold}')
    print(f'Weights path: {weights_path}')

    detector = build_detector(config)
    descriptor = build_descriptor(config)

    model = TraqPoint(
        detector,
        descriptor,
        detection_threshold=config['detection_threshold'],
        top_k=config['top_k'],
        train_detector=config['train_detector'],
        device=torch.device(config['device'])
    )

    if os.path.exists(weights_path):
        print(f"Loading weights from: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location='cpu'), strict=True)
    else:
        print(f"Warning: Weights file not found at {weights_path}")

    model.to(torch.device(config['device']))
    return model


# Import trajectory tracking modules
try:
    from benchmarks.visual_odometry.track_utils.tracker import TrajectoryTracker, TrajectoryEvaluator
    TRACKING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Trajectory tracking not available: {e}")
    TRACKING_AVAILABLE = False

vo_config = {
    'dataset': {
        'name': 'KITTILoader',
        'root_path': '/home/yepeng_liu/code_python/dataset/visual_odometry/kitty/gray',
        'sequence': '10',
        'start': 0
    },
    'detector': {
        'name': 'LiftFeatDetector',
        'descriptor_dim': 64,
        'nms_radius': 5,
        'keypoint_threshold': 0.005,
        'max_keypoints': 4096,
        'remove_borders': 4,
        'cuda': 1
    },
    'matcher': {
        'name': 'FrameByFrameMatcher',
        'type': 'FLANN',
        'FLANN': {
            'kdTrees': 5,
            'searchChecks': 50
        },
        'distance_ratio': 0.85
    },
    'tracking': {
        'enabled': True,              # Enable trajectory tracking
        'distance_threshold': 30.0,    # Trajectory tracking distance threshold
        'max_missing_frames': 1,      # Maximum missing frames
        'batch_size': 4               # Batch size
    }
}

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def keypoints_plot(img, vo, img_id, path2):
    img_ = cv2.imread(path2+str(img_id-1).zfill(6)+".png")
  
    if not vo.match_kps:
        img_ = plot_keypoints(img_, vo.kptdescs["cur"]["keypoints"])
    else:
        for index in range(vo.match_kps["ref"].shape[0]):
            ref_point = tuple(map(int, vo.match_kps['ref'][index,:]))  # Convert keypoints to integer tuples
            cur_point = tuple(map(int, vo.match_kps['cur'][index,:]))
            cv2.line(img_, ref_point, cur_point, (0, 255, 0), 2)  # Draw green line
            cv2.circle(img_, cur_point, 3, (0, 0, 255), -1)  # Draw red circle at current keypoint

    return img_

class TrajPlotter(object):
    def __init__(self):
        self.errors = []
        self.traj = np.zeros((800, 800, 3), dtype=np.uint8)

    def update(self, est_xyz, gt_xyz):
        x, z = est_xyz[0], est_xyz[2]
        gt_x, gt_z = gt_xyz[0], gt_xyz[2]
        est = np.array([x, z]).reshape(2)
        gt = np.array([gt_x, gt_z]).reshape(2)
        error = np.linalg.norm(est - gt)
        self.errors.append(error)
        avg_error = np.mean(np.array(self.errors))
        draw_x, draw_y = int(x) + 80, int(z) + 230
        true_x, true_y = int(gt_x) + 80, int(gt_z) + 230
        cv2.circle(self.traj, (draw_x, draw_y), 1, (0, 0, 255), 1)
        cv2.circle(self.traj, (true_x, true_y), 1, (0, 255, 0), 2)
        cv2.rectangle(self.traj, (10, 5), (450, 120), (0, 0, 0), -1)
        text = "[AvgError] %2.4fm" % (avg_error)
        print(text)
        cv2.putText(self.traj, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        note = "Green: GT, Red: Predict"
        cv2.putText(self.traj, note, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return self.traj

class EnhancedTrajPlotter(object):
    def __init__(self, tracking_enabled=False, tracking_config=None):
        self.errors = []
        self.traj = np.zeros((800, 800, 3), dtype=np.uint8)
        self.frame_count = 0

        # Trajectory tracking related
        self.tracking_enabled = tracking_enabled and TRACKING_AVAILABLE
        self.tracker = None
        self.evaluator = None
        self.tracking_stats = {}

        if self.tracking_enabled and tracking_config:
            self.tracker = TrajectoryTracker(
                distance_threshold=tracking_config.get('distance_threshold', 30.0),
                max_missing_frames=tracking_config.get('max_missing_frames', 3)
            )
            self.evaluator = TrajectoryEvaluator()
            print("Trajectory tracking enabled")

    def update(self, est_xyz, gt_xyz, detected_keypoints=None, detected_des=None):
        self.frame_count += 1
        x, z = est_xyz[0], est_xyz[2]
        gt_x, gt_z = gt_xyz[0], gt_xyz[2]
        est = np.array([x, z]).reshape(2)
        gt = np.array([gt_x, gt_z]).reshape(2)
        error = np.linalg.norm(est - gt)
        self.errors.append(error)
        avg_error = np.mean(np.array(self.errors))
        
        if self.tracking_enabled and detected_keypoints is not None:
            try:
                tracking_result = self.tracker.update(detected_keypoints, detected_des, self.frame_count)
                self.evaluator.add_ground_truth(detected_keypoints, self.frame_count)
                self.evaluator.add_estimated(detected_keypoints, self.frame_count)
                self.tracking_stats = self.tracker.get_statistics()
            except Exception as e:
                print(f"Trajectory tracking update failed: {e}")

        draw_x, draw_y = int(x) + 80, int(z) + 230
        true_x, true_y = int(gt_x) + 80, int(gt_z) + 230

        cv2.circle(self.traj, (draw_x, draw_y), 1, (0, 0, 255), 1)
        cv2.circle(self.traj, (true_x, true_y), 1, (0, 255, 0), 2)
        cv2.rectangle(self.traj, (10, 5), (500, 200), (0, 0, 0), -1)
        text = "[AvgError] %2.4fm" % (avg_error)
        cv2.putText(self.traj, text, (20, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        if self.tracking_enabled and self.tracking_stats:
            y_offset = 85
            cv2.putText(self.traj, "Trajectory Tracking:", (20, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
            y_offset += 25
            cv2.putText(self.traj, "Active Tracks: {}".format(self.tracking_stats.get('active_trackers', 0)),
                          (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 20
            cv2.putText(self.traj, "Avg Length: {:.1f}".format(self.tracking_stats.get('average_trajectory_length', 0)),
                          (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 20
            cv2.putText(self.traj, "Total Created: {}".format(self.tracking_stats.get('total_trackers_created', 0)),
                          (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 20
            cv2.putText(self.traj, "Avg Displacement: {:.1f}".format(self.tracking_stats.get('average_displacement', 0)),
                          (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(self.traj, "Green: GT, Red: Predict", (20, 180),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return self.traj

    def get_evaluation_results(self):
        """Get evaluation results"""
        results = {
            'position_error': {
                'mean_error': np.mean(self.errors) if self.errors else 0,
                'std_error': np.std(self.errors) if self.errors else 0,
                'max_error': np.max(self.errors) if self.errors else 0,
                'min_error': np.min(self.errors) if self.errors else 0,
                'rmse': np.sqrt(np.mean(np.array(self.errors)**2)) if self.errors else 0
            },
            'frame_count': self.frame_count
        }
        
        # Add trajectory tracking evaluation results
        if self.tracking_enabled and self.evaluator:
            try:
                ate_results = self.evaluator.compute_absolute_trajectory_error()
                if 'error' not in ate_results:
                    results['absolute_trajectory_error'] = ate_results
                
                trajectories = self.tracker.get_all_trajectories()
                length_stats = self.evaluator.compute_trajectory_length_statistics(trajectories)
                if 'error' not in length_stats:
                    results['trajectory_length_statistics'] = length_stats
                    
            except Exception as e:
                print(f"Failed to get trajectory evaluation results: {e}")

        # Add ATE based on camera poses (no trajectory tracking required)
        if hasattr(self, 'camera_poses') and len(self.camera_poses) > 0:
            try:
                camera_ate = self.compute_camera_ate()
                if camera_ate:
                    results['camera_absolute_trajectory_error'] = camera_ate
            except Exception as e:
                print(f"Failed to compute camera pose ATE: {e}")
        
        return results




def run_video(args):
    # create dataloader
    vo_config["dataset"]['root_path'] = args.path1
    vo_config["dataset"]['sequence'] = args.id
    # Update parallel processing configuration

    dir_out = args.out_dir
    # Check trajectory tracking functionality
    tracking_enabled = args.enable_tracking and TRACKING_AVAILABLE
    if args.enable_tracking and not TRACKING_AVAILABLE:
        print("Warning: Trajectory tracking not available, disabling this feature")

    print(f"Trajectory tracking: {'Enabled' if tracking_enabled else 'Disabled'}")
    if tracking_enabled:
        print(f"Distance threshold: {args.distance_threshold}")
        print(f"Max missing frames: {args.max_missing_frames}")



    loader = create_dataloader(vo_config["dataset"])
    model = build_vo_model(detection_threshold=args.det_th)
    model.eval()
    detector = model
    # create matcher
    matcher = create_matcher(vo_config["matcher"])

    absscale = AbosluteScaleComputer()
    vo = VisualOdometry(detector, matcher, loader.cam)
    # Create enhanced trajectory plotter
    tracking_config = {
        'distance_threshold': args.distance_threshold,
        'max_missing_frames': args.max_missing_frames
    }
    traj_plotter = EnhancedTrajPlotter(tracking_enabled, tracking_config)

    
    # Create output directory
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    
    # Generate filename
    tracking_suffix = "_with_tracking" if tracking_enabled else ""
    fname = f"kitti_{vo_config['detector']['name']}_{vo_config['matcher']['type']}match{tracking_suffix}"
    
    # Create log file
    log_fopen = open(dir_out + "/" + fname + ".txt", mode='w')
    
    # Create video writer
    keypoints_video_path = dir_out + "/" + fname + "_keypoints.avi"
    trajectory_video_path = dir_out + "/" + fname + "_trajectory.avi"

    # Set up video writer: choose codec and set FPS and frame size
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 10  # Adjust the FPS according to your input data
    frame_size = (1200, 400)  # Get frame size from first image

    # Video writers for keypoints and trajectory
    keypoints_writer = cv2.VideoWriter(keypoints_video_path, fourcc, fps, frame_size)
    trajectory_writer = cv2.VideoWriter(trajectory_video_path, fourcc, fps, (800, 800))
    
    # Process frames
    frame_count = 0

    # Serial processing mode
    print("Using serial processing mode")
    for i, img in enumerate(loader):
            img_id = loader.img_id
            gt_pose = loader.get_cur_pose()

            # Update visual odometry
            R, t = vo.update(img, absscale.update(gt_pose))

            # Get detected keypoints
            detected_keypoints = None
            detected_des = None
            if hasattr(vo, 'kptdescs') and 'cur' in vo.kptdescs:
                detected_keypoints = vo.kptdescs['cur']['keypoints']
                detected_des = vo.kptdescs['cur']['descriptors']

            # Generate visualization
            img1 = keypoints_plot(img, vo, img_id, args.path2)
            img1 = cv2.resize(img1, (1200, 400))
            img2 = traj_plotter.update(t, gt_pose[:, 3], detected_keypoints, detected_des)

            # Write to log
            print(i, t[0, 0], t[1, 0], t[2, 0], gt_pose[0, 3], gt_pose[1, 3], gt_pose[2, 3], file=log_fopen)

            # Write to video
            keypoints_writer.write(img1)
            trajectory_writer.write(img2)

            frame_count += 1

            # Print progress
            if frame_count % 10 == 0 or frame_count <= 5:
                print(f"Processed frame {i}")

            # # Only run first 50 frames for comparison
            # if frame_count >= 50:
            #     break
        
   
    # Clean up resources (keep)
    log_fopen.close()
    keypoints_writer.release()  # Note: original code uses release(), OpenCV-python should use close()
    trajectory_writer.release()
    
    print(f"Processing completed!")
    print(f"Total frames: {frame_count}")
    print(f"Videos saved: {keypoints_video_path} and {trajectory_video_path}")

    # Get and save evaluation results
    evaluation_results = traj_plotter.get_evaluation_results()
    evaluation_results['processing_info'] = {
        'total_frames': frame_count,
        'tracking_enabled': tracking_enabled
    }

    # Save results to JSON file
    results_file = dir_out + "/" + f"{fname}_results.json"
    # Note: original code doesn't import json, need to add import
    import json
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    print(f"Evaluation results saved: {results_file}")

    # Print results summary
    print_results_summary(evaluation_results)
   
def print_results_summary(results):
    """Print results summary"""
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)

    # Position error
    pos_error = results.get('position_error', {})
    if pos_error:
        print("Position error statistics:")
        print(f"  Mean error: {pos_error.get('mean_error', 0):.4f}m")
        print(f"  RMSE: {pos_error.get('rmse', 0):.4f}m")
        print(f"  Max error: {pos_error.get('max_error', 0):.4f}m")
        print(f"  Min error: {pos_error.get('min_error', 0):.4f}m")
        print(f"  Std dev: {pos_error.get('std_error', 0):.4f}m")

    # Trajectory length statistics
    length_stats = results.get('trajectory_length_statistics', {})
    if length_stats and 'error' not in length_stats:
        print("\nTrajectory length statistics:")
        print(f"  Average trajectory length: {length_stats.get('average_length', 0):.2f}")
        print(f"  Max trajectory length: {length_stats.get('max_length', 0)}")
        print(f"  Min trajectory length: {length_stats.get('min_length', 0)}")
        print(f"  Total trajectories: {length_stats.get('total_trajectories', 0)}")

    # Absolute trajectory error
    ate = results.get('absolute_trajectory_error', {})
    if ate and 'error' not in ate:
        print("\nAbsolute trajectory error (ATE):")
        print(f"  Mean error: {ate.get('mean_error', 0):.4f}")
        print(f"  RMSE: {ate.get('rmse', 0):.4f}")
        print(f"  Max error: {ate.get('max_error', 0):.4f}")
        print(f"  Min error: {ate.get('min_error', 0):.4f}")

    # Processing information
    proc_info = results.get('processing_info', {})
    if proc_info:
        print("\nProcessing information:")
        print(f"  Total frames: {proc_info.get('total_frames', 0)}")
        print(f"  Trajectory tracking: {'Enabled' if proc_info.get('tracking_enabled', False) else 'Disabled'}")
        print(f"  Parallel processing: {'Enabled' if proc_info.get('parallel_enabled', False) else 'Disabled'}")
        if proc_info.get('parallel_enabled', False):
            print(f"  Parallel workers: {proc_info.get('parallel_workers', 0)}")

    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='python_vo')
   
    parser.add_argument('--path1', type=str, default='/high_perf_store3/evad-autolabeling/lihao_data/lyp-data/kitty/dataset',
                        help='KITTI dataset root path')
    parser.add_argument('--path2', type=str, default="/high_perf_store3/evad-autolabeling/lihao_data/lyp-data/kitty/dataset/sequences/01/image_0/",
                        help='KITTI image sequence path')
    parser.add_argument('--id', type=str, default="01",
                        help='config file')
    parser.add_argument('--out_dir', type=str, default="./outputs/vo_kitty_results",
                        help='Output directory for VO results')
   
    # Trajectory tracking parameters
    parser.add_argument('--enable_tracking', action='store_true', default=True,
                       help='Enable trajectory tracking feature')
    parser.add_argument('--distance_threshold', type=float, default=30.0,
                       help='Trajectory tracking distance threshold')
    parser.add_argument('--max_missing_frames', type=int, default=1,
                       help='Maximum missing frames')


    # Other parameters
    parser.add_argument('--det_th', type=float, default=0.5,
                       help='Feature detection threshold')
    parser.add_argument('--max_frames', type=int, default=50,
                       help='Maximum processing frames (for testing, default 50)')

    args = parser.parse_args()

    run_video(args)
