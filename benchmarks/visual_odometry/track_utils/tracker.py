import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import logging

class Tracker:
    def __init__(self, point_id: int, initial_point: np.ndarray, initial_descriptor: np.ndarray, frame_id: int):
        self.point_id = point_id
        self.trajectory = [initial_point.copy()]
        self.frame_ids = [frame_id]
        self.track_success_count = 1
        self.is_alive = True
        self.last_seen_frame = frame_id
        self.latest_descriptor = initial_descriptor.copy()

    def update(self, new_point: np.ndarray, new_descriptor: np.ndarray, frame_id: int):
        self.trajectory.append(new_point.copy())
        self.latest_descriptor=new_descriptor.copy()
        self.frame_ids.append(frame_id)
        self.track_success_count += 1
        self.last_seen_frame = frame_id

    def get_length(self) -> int:
        return len(self.trajectory)
    
    def get_trajectory_points(self) -> List[np.ndarray]:
        return self.trajectory.copy()

    def die(self):
        self.is_alive = False

    def get_displacement(self) -> float:
        if len(self.trajectory) < 2:
            return 0.0
        start_point = self.trajectory[0]
        end_point = self.trajectory[-1]
        return np.linalg.norm(end_point - start_point)

class TrajectoryTracker:
    def __init__(self, distance_threshold: float = 30.0, max_missing_frames: int = 3):
        self.distance_threshold = distance_threshold
        self.max_missing_frames = max_missing_frames
        self.trackers: List[Tracker] = []
        self.next_point_id = 0
        self.current_frame_id = 0
        self.total_trackers_created = 0
        self.total_trackers_died = 0
        
    def update(self, detected_points: np.ndarray, detected_descriptors: np.ndarray, frame_id: int) -> Dict:
        self.current_frame_id = frame_id
        new_trackers = []
        matched_indices = []
        active_mask = np.array([tracker.is_alive for tracker in self.trackers])
        active_trackers = [t for t, is_alive in zip(self.trackers, active_mask) if is_alive]
        if not active_trackers:
            for point, desc in zip(detected_points, detected_descriptors):
                new_tracker = Tracker(self.next_point_id, point, desc, frame_id)
                self.trackers.append(new_tracker)
                new_trackers.append(new_tracker)
                self.next_point_id += 1
                self.total_trackers_created += 1
            self.trackers = [t for t in self.trackers if t.is_alive]
            return {
                'new_trackers': new_trackers,
                'matched_indices': matched_indices,
                'active_trackers': self.trackers,
                'total_trackers': len(self.trackers)
            }
        active_last_points_np = np.array([t.trajectory[-1] for t in active_trackers])
        active_last_descs_np = np.array([t.latest_descriptor for t in active_trackers])
        active_original_indices = [idx for idx, is_alive in zip(range(len(self.trackers)), active_mask) if is_alive]
        matched_flags = np.zeros(len(active_trackers), dtype=bool)
        for i, (point, desc) in enumerate(zip(detected_points, detected_descriptors)):
            unmatched_points = active_last_points_np[~matched_flags]
            if len(unmatched_points) == 0:
                pass
            else:
                distances = np.linalg.norm(unmatched_points - point, axis=1)
                min_dist = np.min(distances)
                if min_dist < self.distance_threshold:
                    min_idx_in_unmatched = np.argmin(distances)
                    min_idx_in_active = np.where(~matched_flags)[0][min_idx_in_unmatched]
                    candidate_desc = active_last_descs_np[min_idx_in_active]
                    norm1 = np.linalg.norm(candidate_desc)
                    norm2 = np.linalg.norm(desc)
                    if norm1 < 1e-6 or norm2 < 1e-6:
                        cos_sim = 0.0
                    else:
                        cos_sim = np.dot(candidate_desc/norm1, desc/norm2)
                    if cos_sim > 0.5:
                        original_tracker_idx = active_original_indices[min_idx_in_active]
                        self.trackers[original_tracker_idx].update(point, desc, frame_id)
                        matched_flags[min_idx_in_active] = True
                        matched_indices.append(i)
                        continue
            new_tracker = Tracker(self.next_point_id, point, desc, frame_id)
            self.trackers.append(new_tracker)
            new_trackers.append(new_tracker)
            self.next_point_id += 1
            self.total_trackers_created += 1
        alive_trackers = []
        for tracker in self.trackers:
            if tracker.is_alive and (frame_id - tracker.last_seen_frame) <= self.max_missing_frames:
                alive_trackers.append(tracker)
            else:
                if tracker.is_alive:
                    self.total_trackers_died += 1
                    tracker.die()
        self.trackers = alive_trackers
        return {
            'new_trackers': new_trackers,
            'matched_indices': matched_indices,
            'active_trackers': self.trackers,
            'total_trackers': len(self.trackers)
        }
    
    def _find_best_match(self, point: np.ndarray) -> Optional[int]:
        best_match_idx = None
        min_distance = float('inf')
        for i, tracker in enumerate(self.trackers):
            if not tracker.is_alive:
                continue
            last_point = tracker.trajectory[-1]
            distance = np.linalg.norm(point - last_point)
            if distance < self.distance_threshold and distance < min_distance:
                min_distance = distance
                best_match_idx = i
        return best_match_idx

    def get_statistics(self) -> Dict:
        alive_trackers = [t for t in self.trackers if t.is_alive]
        if not alive_trackers:
            return {
                'total_trackers_created': self.total_trackers_created,
                'total_trackers_died': self.total_trackers_died,
                'active_trackers': 0,
                'average_trajectory_length': 0.0,
                'average_track_success_count': 0.0,
                'average_displacement': 0.0
            }
        trajectory_lengths = [t.get_length() for t in alive_trackers]
        average_trajectory_length = np.mean(trajectory_lengths)
        track_success_counts = [t.track_success_count for t in alive_trackers]
        average_track_success_count = np.mean(track_success_counts)
        displacements = [t.get_displacement() for t in alive_trackers]
        average_displacement = np.mean(displacements)
        return {
            'total_trackers_created': self.total_trackers_created,
            'total_trackers_died': self.total_trackers_died,
            'active_trackers': len(alive_trackers),
            'average_trajectory_length': average_trajectory_length,
            'average_track_success_count': average_track_success_count,
            'average_displacement': average_displacement
        }

    def get_all_trajectories(self) -> List[List[np.ndarray]]:
        return [t.get_trajectory_points() for t in self.trackers if t.is_alive]

    def visualize_trajectories(self, image: np.ndarray, max_trajectories: int = 50) -> np.ndarray:
        vis_image = image.copy()
        alive_trackers = [t for t in self.trackers if t.is_alive]
        if len(alive_trackers) > max_trajectories:
            alive_trackers = alive_trackers[:max_trajectories]
        colors = self._generate_colors(len(alive_trackers))
        for i, tracker in enumerate(alive_trackers):
            color = colors[i]
            trajectory = tracker.get_trajectory_points()
            for j in range(1, len(trajectory)):
                pt1 = tuple(map(int, trajectory[j-1]))
                pt2 = tuple(map(int, trajectory[j]))
                cv2.line(vis_image, pt1, pt2, color, 2)
            if trajectory:
                current_point = tuple(map(int, trajectory[-1]))
                cv2.circle(vis_image, current_point, 5, color, -1)
                cv2.circle(vis_image, current_point, 8, (255, 255, 255), 2)
        return vis_image

    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        colors = []
        for i in range(num_colors):
            hue = int(180 * i / num_colors)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))
        return colors

class TrajectoryEvaluator:
    def __init__(self):
        self.ground_truth_trajectories = []
        self.estimated_trajectories = []
        self.frame_errors = []

    def add_ground_truth(self, gt_points: np.ndarray, frame_id: int):
        self.ground_truth_trajectories.append({
            'points': gt_points.copy(),
            'frame_id': frame_id
        })

    def add_estimated(self, est_points: np.ndarray, frame_id: int):
        self.estimated_trajectories.append({
            'points': est_points.copy(),
            'frame_id': frame_id
        })

    def compute_absolute_trajectory_error(self) -> Dict:
        if len(self.ground_truth_trajectories) != len(self.estimated_trajectories):
            logging.warning("Ground truth and estimated trajectories length mismatch")
            return {'error': 'Length mismatch'}
        errors = []
        for gt_data, est_data in zip(self.ground_truth_trajectories, self.estimated_trajectories):
            gt_points = gt_data['points']
            est_points = est_data['points']
            if len(gt_points) != len(est_points):
                continue
            point_errors = np.linalg.norm(gt_points - est_points, axis=1)
            errors.extend(point_errors)
        if not errors:
            return {'error': 'No valid data'}
        errors = np.array(errors)
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'median_error': np.median(errors),
            'rmse': np.sqrt(np.mean(errors**2))
        }

    def compute_trajectory_length_statistics(self, trajectories: List[List[np.ndarray]]) -> Dict:
        if not trajectories:
            return {'error': 'No trajectories'}
        lengths = [len(traj) for traj in trajectories]
        return {
            'average_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'max_length': np.max(lengths),
            'min_length': np.min(lengths),
            'median_length': np.median(lengths),
            'total_trajectories': len(trajectories)
        }

    def reset(self):
        self.ground_truth_trajectories = []
        self.estimated_trajectories = []
        self.frame_errors = []
