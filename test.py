#!/usr/bin/env python3
"""Motion planning for autonomous riderless bike using monocular vision."""

import argparse
import logging
import cv2
import rerun as rr
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor
from collections import deque
from typing import Dict, Tuple, List

# TODO: FIX THIS
def create_arm_mask(width: int, height: int) -> np.ndarray:
    """
    Create a mask to ignore upper side regions where arms typically appear (only for stock video of biker's pov)

    """
    mask = np.ones((height, width), dtype=bool)
    
    # Define the regions to mask based on empirical observations of common false positives.
    corner_height = height * 2 // 3  # Focus on the upper two-thirds where arms are likely.
    start_height = height // 6       # Avoid starting too close to the top.
    corner_width = width // 4        # Focus on the outer edges.

    mask[start_height:corner_height, :corner_width] = False
    mask[start_height:corner_height, -corner_width:] = False
    
    # rr.log(
    #     "arm_mask",
    #     rr.Image(mask.astype(np.uint8) * 255)
    # )
    
    return mask

class ObjectTracker:
    """
    Tracks objects across frames and estimates their velocity.

    - Maintain a track of objects for velocity estimation which is used for predicting object motion
    - Using a deque is more efficient for of a fixed-size memory for past object states
    """
    def __init__(self, memory_size: int = 10):  # Larger memory allows for smoother velocity estimation
        self.tracks: Dict[int, deque] = {}
        self.next_track_id = 0
        self.memory_size = memory_size
        self.current_positions: Dict[int, np.ndarray] = {}
        self.filtered_velocities: Dict[int, np.ndarray] = {}  # Store smoothed velocities.

    def update(self, detections: np.ndarray, frame_idx: int) -> Dict[int, np.ndarray]:
        """
        Update tracks with new detections and return velocity estimates

        - By matching new detections to existing tracks continuity is ensured
        - IoU (Intersection over Union) - accounts for spatial overlap
        """
        if len(self.tracks) == 0:
            # Initialize tracks for the first frame. No matching is required here.
            for det in detections:
                self.tracks[self.next_track_id] = deque([(frame_idx, *det[:4])], maxlen=self.memory_size)
                self.current_positions[self.next_track_id] = det[:4]
                self.next_track_id += 1
            return {}
        
        matched_tracks = {}
        unmatched_detections = []

        for det in detections:
            best_iou = 0.3  # Empirical threshold (I just chose this number) balances avoiding false matches and missing real objects
            best_track_id = None
            
            for track_id, track in self.tracks.items():
                if track:  # Checks track has a valid history
                    iou = self._calculate_iou(det[:4], np.array(track[-1][1:5]))
                    if iou > best_iou:
                        best_iou = iou
                        best_track_id = track_id
            
            if best_track_id is not None:
                matched_tracks[best_track_id] = det
            else:
                unmatched_detections.append(det)
        
        # Update matched tracks with the new detections
        velocities = {}
        self.current_positions.clear()
        for track_id, det in matched_tracks.items():
            self.tracks[track_id].append((frame_idx, *det[:4]))
            self.current_positions[track_id] = det[:4]
            if len(self.tracks[track_id]) >= 2:
                velocities[track_id] = self._estimate_velocity(self.tracks[track_id])
        
        # Create new tracks for unmatched detections, treating them as new objects
        for det in unmatched_detections:
            self.tracks[self.next_track_id] = deque([(frame_idx, *det[:4])], maxlen=self.memory_size)
            self.current_positions[self.next_track_id] = det[:4]
            self.next_track_id += 1
        
        return velocities

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate IoU between two bounding boxes

        - IoU captures the overlap ratio
        """
        box1_x2 = box1[0] + box1[2]
        box1_y2 = box1[1] + box1[3]
        box2_x2 = box2[0] + box2[2]
        box2_y2 = box2[1] + box2[3]
        
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1_x2, box2_x2)
        yi2 = min(box1_y2, box2_y2)
        
        intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0

    def _estimate_velocity(self, track: deque) -> np.ndarray:
        """
        Estimate velocity with smoothing for reliability.

        - Velocity estimation requires multiple observations
        - Using smoothing (median and exponential) removes noise
        """
        if len(track) < 2:
            return np.zeros(2)
        
        track_array = np.array(list(track))
        frames = track_array[:, 0]
        positions = np.array([[x + w/2, y + h/2] for _, x, y, w, h in track_array])
        
        # Compute velocities between consecutive observations
        velocities = []
        for i in range(len(positions) - 1):
            dt = frames[i+1] - frames[i]
            if dt > 0:  # Ensure valid time intervals.
                vel = (positions[i+1] - positions[i]) / dt
                velocities.append(vel)
        
        if not velocities:
            return np.zeros(2)
        
        # Median velocity is robust to outliers
        median_velocity = np.median(velocities, axis=0)
        
        # Smooths velocity using exponential smoothing
        track_id = id(track)  # Track ID
        if track_id not in self.filtered_velocities:
            self.filtered_velocities[track_id] = median_velocity
        else:
            alpha = 0.3  # Smoothing factor for responsiveness and stability
            self.filtered_velocities[track_id] = (
                alpha * median_velocity + 
                (1 - alpha) * self.filtered_velocities[track_id]
            )
        
        # Apply a minimum velocity threshold to filter out insignificant movement
        min_velocity_threshold = 0.5
        if np.linalg.norm(self.filtered_velocities[track_id]) < min_velocity_threshold:
            return np.zeros(2)
            
        return self.filtered_velocities[track_id]

class MotionPlanner:
    """
    Plans motion for the bike based on object detections and depth information

    """
    def __init__(self, image_width: int, image_height: int):
        self.image_width = image_width
        self.image_height = image_height
        self.min_safe_distance = 2.0  # Minimum distance considered safe
        self.danger_distance = 3.0   # Distance at which to begin avoidance
        self.critical_distance = 1.0  # Distance below which an emergency stop is required
        
    def get_object_distance(self, detection: np.ndarray, depth_map: np.ndarray) -> float:
        """
        Estimate the distance to an object using a depth map

        - Median depth values within the bounding box are used to reduce the influence of outliers
        """
        x, y, w, h = detection[:4]
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        x = max(0, min(x, depth_map.shape[1] - 1))
        y = max(0, min(y, depth_map.shape[0] - 1))
        w = min(w, depth_map.shape[1] - x)
        h = min(h, depth_map.shape[0] - y)
        
        # Extract the region of interest (ROI) in the depth map
        roi = depth_map[y:y+h, x:x+w]
        
        if roi.size == 0:
            return float('inf')  # Infinite distance if no valid depth
        
        return np.median(roi)
        
    def plan_motion(self, detections: np.ndarray, velocities: Dict[int, np.ndarray], 
                    depth_map: np.ndarray) -> Tuple[str, float]:
        """
        Plan the car's motion based on detected objects and their distances

        - Depth map regions are compared to identify the safest direction to turn
        """
        if len(detections) == 0:
            return 'forward', 0.0  # No obstacles; proceed forward.
        
        object_distances = []
        for det in detections:
            distance = self.get_object_distance(det, depth_map)
            object_distances.append(distance)
        
        # Emergency stop if any object is within a critical distance.
        if any(d < self.critical_distance for d in object_distances):
            return 'stop', 0.0
            
        # Find the closest object and its position.
        closest_idx = np.argmin(object_distances)
        closest_distance = object_distances[closest_idx]
        closest_obj = detections[closest_idx]
        obj_center_x = closest_obj[0] + closest_obj[2] / 2
        
        # If the closest object is safely far, continue moving
        if closest_distance > self.danger_distance:
            return 'forward', 0.0
            
        # Decide which direction to turn based on object position
        center_x = self.image_width / 2
        turn_angle = 30.0  # Set angle

        # Use depth data to check which side is safer
        left_region = depth_map[:, :int(self.image_width / 3)].mean()
        right_region = depth_map[:, int(2 * self.image_width / 3):].mean()
        
        if obj_center_x < center_x:
            # Object on the left; prefer turning right if safe
            if right_region > self.min_safe_distance:
                return 'turn_right', -turn_angle
        else:
            # Object on the right; prefer turning left if safe
            if left_region > self.min_safe_distance:
                return 'turn_left', turn_angle
                
        # If neither side is clear, stop!
        return 'stop', 0.0

class DepthEstimator:
    """
    Uses a pre-trained MiDaS DPT model to estimate depth from RGB images

    - Using a pre-trained model avoids the need for manual depth calibration (still need the scale factor)
    """
    def __init__(self):
        self.processor = DPTImageProcessor.from_pretrained("Intel/dpt-swinv2-tiny-256") # This is the lightest from intel
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-swinv2-tiny-256")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Real-world scaling factor for converting depth values to meters
        # This value should be calibrated for the specific camera setup
        self.scale_factor = 0.1  # Meters per depth unit
        
    def estimate_depth(self, rgb_image: np.ndarray) -> np.ndarray:
        inputs = self.processor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        depth_map = predicted_depth.squeeze().cpu().numpy()
        depth_map = depth_map * self.scale_factor
        
        return depth_map
    
def process_video(video_source: str, *, max_frame_count: int | None = None) -> None:
    """
    - Sets up and integrates various components (detection, tracking, depth estimation, and planning)
    - logging visualization and debugging via Rerun and terminal
    """
    model = YOLO('yolov8n.pt')
    tracker = ObjectTracker()
    depth_estimator = DepthEstimator()
    
    # video capture (both camera streams and video files - mp4 so far)
    if video_source.isdigit():
        cap = cv2.VideoCapture(int(video_source))
        logging.info("Using camera device %s", video_source)
    else:
        cap = cv2.VideoCapture(video_source)
        logging.info("Loading video file: %s", video_source)
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {video_source}")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    planner = MotionPlanner(frame_width, frame_height)
    
    arm_mask = create_arm_mask(frame_width, frame_height)
    
    frame_idx = 0
    while cap.isOpened():
        if max_frame_count is not None and frame_idx >= max_frame_count:
            break

        ret, bgr = cap.read()
        if not ret:
            logging.info("End of video")
            break

        # Logging
        rr.set_time_sequence("frame", frame_idx)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rr.log("raw_video", rr.Image(rgb).compress(jpeg_quality=85))
        
        depth_map = depth_estimator.estimate_depth(rgb)
        normalized_depth = ((depth_map - depth_map.min()) / 
                            (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_TURBO)
        rr.log("depth_map", rr.Image(depth_colored))
        
        # Run object detection and filter results based on relevance
        results = model(rgb)
        all_detections = []
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                relevant_classes = [0, 1, 2, 3, 5, 7]  # Only relevant object classes
                mask = np.isin(boxes.cls.cpu().numpy(), relevant_classes)
                
                if np.any(mask):
                    xyxy = boxes.xyxy.cpu().numpy()[mask]
                    valid_detections = []
                    for box in xyxy:
                        center_x = (box[0] + box[2]) / 2
                        center_y = (box[1] + box[3]) / 2
                        
                        # Exclude detections in the arm regions
                        if center_y < frame_height or arm_mask[int(center_y), int(center_x)]:
                            valid_detections.append(box)
                    
                    if valid_detections:
                        valid_detections = np.array(valid_detections)
                        xywh = np.column_stack((
                            valid_detections[:, 0],
                            valid_detections[:, 1],
                            valid_detections[:, 2] - valid_detections[:, 0],
                            valid_detections[:, 3] - valid_detections[:, 1]
                        ))
                        class_ids = boxes.cls.cpu().numpy()[mask][:len(valid_detections)]
                        detections = np.column_stack((xywh, class_ids))
                        all_detections.extend(detections)
        
        if all_detections:
            all_detections = np.array(all_detections)
            velocities = tracker.update(all_detections, frame_idx)
            action, turn_angle = planner.plan_motion(all_detections, velocities, depth_map)
            
            # Log planning decision
            rr.log("planning", rr.TextLog(f"Action: {action}, Turn Angle: {turn_angle:.1f}°"))
            
            # Log bounding boxes (this was missing)
            rr.log(
                "detections",
                rr.Boxes2D(
                    array=all_detections[:, :4],
                    array_format=rr.Box2DFormat.XYWH,
                    class_ids=all_detections[:, -1].astype(np.uint16),
                )
            )
            
            # Also log velocity vectors if you want them
            for track_id, vel in velocities.items():
                if track_id in tracker.current_positions and np.linalg.norm(vel) > 0:
                    pos = tracker.current_positions[track_id]
                    scaled_vel = vel * 10
                    rr.log(
                        "velocities",
                        rr.Arrows2D(
                            origins=np.array([[pos[0], pos[1]]]),
                            vectors=np.array([scaled_vel]),
                            colors=np.array([[255, 0, 0]]),
                        )
                    )
        
        frame_idx += 1

    cap.release()

def main() -> None:
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Autonomous car motion planning using monocular vision.")
    parser.add_argument(
        "--video-path",
        type=str,
        default="0",
        help="Path to video file or camera device number (default: 0)",
    )
    parser.add_argument(
        "--max-frame",
        type=int,
        help="Stop after processing this many frames",
    )
    parser.add_argument(
        "--min-safe-distance",
        type=float,
        default=2.0,
        help="Minimum safe distance in meters (default: 2.0)",
    )
    parser.add_argument(
        "--danger-distance",
        type=float,
        default=3.0,
        help="Distance at which to start avoidance in meters (default: 3.0)",
    )
    
    rr.script_add_args(parser)
    args = parser.parse_args()

    rr.script_setup(args, "autonomous_car_planning")
    process_video(args.video_path, max_frame_count=args.max_frame)
    rr.script_teardown(args)

if __name__ == "__main__":
    main()