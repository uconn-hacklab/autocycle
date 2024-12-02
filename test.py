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
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum

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


class BikeState(Enum):
    FORWARD = "forward"
    TURNING_LEFT = "turning_left"
    TURNING_RIGHT = "turning_right"
    STOPPING = "stopping"
    STOPPED = "stopped"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class ObstacleData:
    position: np.ndarray  # [x, y]
    velocity: np.ndarray  # [vx, vy]
    size: np.ndarray      # [width, height]
    distance: float
    ttc: float
    predicted_positions: List[np.ndarray]

class EnhancedMotionPlanner:
    def __init__(self, image_width: int, image_height: int):
        self.image_width = image_width
        self.image_height = image_height
        
        # Safety thresholds
        self.min_safe_distance = 2.0   # meters
        self.danger_distance = 3.0     # meters
        self.critical_distance = 1.0   # meters
        
        # Bike parameters
        self.bike_speed = 1.0          # meters/second
        self.max_turn_angle = 35.0     # degrees
        self.min_turn_angle = 0      # degrees
        
        # State machine
        self.current_state = BikeState.FORWARD
        self.target_turn_angle = 0.0
        self.current_turn_angle = 0.0
        self.turn_rate = 5.0           # degrees per frame
        
        # Prediction parameters
        self.prediction_horizon = 10    # frames
        self.prediction_dt = 0.1       # seconds

    def get_object_distance(self, detection: np.ndarray, depth_map: np.ndarray) -> float:
        """Get the distance to an object using the depth map with debugging."""
        x, y, w, h = detection[:4]
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Debug original values
        logging.debug(f"Original bbox: x={x}, y={y}, w={w}, h={h}")
        logging.debug(f"Depth map shape: {depth_map.shape}")
        
        # Ensure coordinates are within bounds
        x = max(0, min(x, depth_map.shape[1] - 1))
        y = max(0, min(y, depth_map.shape[0] - 1))
        w = min(w, depth_map.shape[1] - x)
        h = min(h, depth_map.shape[0] - y)
        
        # Debug adjusted values
        logging.debug(f"Adjusted bbox: x={x}, y={y}, w={w}, h={h}")
        
        # Get the region of interest in the depth map
        roi = depth_map[y:y+h, x:x+w]
        
        if roi.size == 0:
            return float('inf')
        
        # Debug ROI stats
        min_depth = np.min(roi)
        max_depth = np.max(roi)
        median_depth = np.median(roi)
        mean_depth = np.mean(roi)
        
        return median_depth
        
    def estimate_ttc(self, obstacle_pos: np.ndarray, obstacle_vel: np.ndarray, 
                     distance: float) -> float:
        """Estimate time-to-collision with an obstacle."""
        # Convert pixel velocities to approximate real-world velocities using distance
        scale_factor = distance / 10.0  # rough approximation
        relative_vel = obstacle_vel * scale_factor - np.array([0, self.bike_speed])
        
        # If moving away or parallel, return infinite TTC
        if np.dot(relative_vel, obstacle_pos) >= 0:
            return float('inf')
            
        # Calculate TTC
        relative_speed = np.linalg.norm(relative_vel)
        if relative_speed < 0.1:  # Practically stationary
            return float('inf')
            
        ttc = distance / relative_speed
        return ttc
    
    def predict_obstacle_path(self, position: np.ndarray, velocity: np.ndarray) -> List[np.ndarray]:
        """Predict future positions of an obstacle."""
        predictions = []
        current_pos = position.copy()
        
        for _ in range(self.prediction_horizon):
            current_pos = current_pos + velocity * self.prediction_dt
            predictions.append(current_pos.copy())
            
        return predictions
    
    def calculate_turn_angle(self, obstacles: List[ObstacleData]) -> float:
        """Calculate smooth turn angle based on obstacles."""
        if not obstacles:
            # Gradually return to forward if no obstacles
            if abs(self.current_turn_angle) < self.min_turn_angle:
                return 0.0
            return self.current_turn_angle * 0.9  # Gradually straighten
            
        # Find the most urgent obstacle
        min_ttc = float('inf')
        urgent_obstacle = None
        
        for obs in obstacles:
            if obs.ttc < min_ttc and obs.distance < self.danger_distance:
                min_ttc = obs.ttc
                urgent_obstacle = obs
                
        if urgent_obstacle is None:
            return 0.0
            
        # Calculate desired turn angle
        obstacle_center = urgent_obstacle.position[0]
        image_center = self.image_width / 2
        
        # Basic angle calculation
        base_angle = 30.0 * (image_center - obstacle_center) / (self.image_width / 2)
        
        # Adjust angle based on TTC
        ttc_factor = max(0.0, min(1.0, (urgent_obstacle.ttc - 1.0) / 2.0))
        target_angle = base_angle * (1.0 - ttc_factor)
        
        # Smooth the turn
        angle_diff = target_angle - self.current_turn_angle
        if abs(angle_diff) > self.turn_rate:
            if angle_diff > 0:
                return self.current_turn_angle + self.turn_rate
            else:
                return self.current_turn_angle - self.turn_rate
                
        return target_angle
    
    def update_state(self, obstacles: List[ObstacleData]) -> Tuple[BikeState, float]:
        """Update state machine and return new state and turn angle."""
        if not obstacles:
            # No obstacles - transition to or maintain FORWARD
            if self.current_state != BikeState.FORWARD:
                self.current_state = BikeState.FORWARD
            self.target_turn_angle = 0.0
            
        else:
            # Check for emergency stops
            for obs in obstacles:
                if obs.distance < self.critical_distance or obs.ttc < 0.5:
                    self.current_state = BikeState.EMERGENCY_STOP
                    return BikeState.EMERGENCY_STOP, 0.0
            
            # Calculate new turn angle
            self.target_turn_angle = self.calculate_turn_angle(obstacles)
            
            # Update state based on turn angle
            if abs(self.target_turn_angle) < self.min_turn_angle:
                self.current_state = BikeState.FORWARD
            elif self.target_turn_angle > 0:
                self.current_state = BikeState.TURNING_LEFT
            else:
                self.current_state = BikeState.TURNING_RIGHT
                
        # Smooth turn angle transition
        angle_diff = self.target_turn_angle - self.current_turn_angle
        if abs(angle_diff) > self.turn_rate:
            if angle_diff > 0:
                self.current_turn_angle += self.turn_rate
            else:
                self.current_turn_angle -= self.turn_rate
        else:
            self.current_turn_angle = self.target_turn_angle
            
        return self.current_state, self.current_turn_angle
    
    def plan_motion(self, detections: np.ndarray, velocities: Dict[int, np.ndarray], 
                    depth_map: np.ndarray) -> Tuple[str, float]:
        """Main motion planning function."""
        obstacles = []
        
        # Process each detection into an obstacle
        for i, det in enumerate(detections):
            position = np.array([det[0] + det[2]/2, det[1] + det[3]/2])
            velocity = velocities.get(i, np.zeros(2))
            size = np.array([det[2], det[3]])
            distance = self.get_object_distance(det, depth_map)
            
            ttc = self.estimate_ttc(position, velocity, distance)
            predicted_positions = self.predict_obstacle_path(position, velocity)
            
            obstacles.append(ObstacleData(
                position=position,
                velocity=velocity,
                size=size,
                distance=distance,
                ttc=ttc,
                predicted_positions=predicted_positions
            ))
            
        # Update state machine
        state, turn_angle = self.update_state(obstacles)
        
        # Log prediction paths if using Rerun
        for obs in obstacles:
            if len(obs.predicted_positions) > 1:
                points = np.array(obs.predicted_positions)
                rr.log(
                    "predictions",
                    rr.Points2D(points),
                    rr.LineStrips2D(points),
                )
        
        return state.value, turn_angle
    
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
        self.scale_factor = 0.0005  # Meters per depth unit
        
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
    """Process video frames for autonomous navigation."""
    model = YOLO('yolov8n.pt')
    tracker = ObjectTracker()
    depth_estimator = DepthEstimator()
    
    # Initialize video capture
    if video_source.isdigit():
        cap = cv2.VideoCapture(int(video_source))
        logging.info("Using camera device %s", video_source)
    else:
        cap = cv2.VideoCapture(video_source)
        logging.info("Loading video file: %s", video_source)
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {video_source}")
    
    # Initialize motion planner with enhanced version
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    planner = EnhancedMotionPlanner(frame_width, frame_height)
    
    frame_idx = 0
    while cap.isOpened():
        if max_frame_count is not None and frame_idx >= max_frame_count:
            break

        ret, bgr = cap.read()
        if not ret:
            logging.info("End of video")
            break

        rr.set_time_sequence("frame", frame_idx)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # Log raw video
        rr.log("raw_video", rr.Image(rgb).compress(jpeg_quality=85))
        
        # Estimate depth
        depth_map = depth_estimator.estimate_depth(rgb)
        normalized_depth = ((depth_map - depth_map.min()) / 
                          (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_TURBO)
        rr.log("depth_map/image", rr.Image(depth_colored))
        
        # Run YOLO detection
        results = model(rgb)
        all_detections = []
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                relevant_classes = [0, 1, 2, 3, 5, 7]
                mask = np.isin(boxes.cls.cpu().numpy(), relevant_classes)
                
                if np.any(mask):
                    xyxy = boxes.xyxy.cpu().numpy()[mask]
                    valid_detections = []
                    for box in xyxy:
                        center_x = (box[0] + box[2]) / 2
                        center_y = (box[1] + box[3]) / 2
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
            
            # Plan motion using enhanced planner
            state, turn_angle = planner.plan_motion(all_detections, velocities, depth_map)
            
            # Log planning visualization
            rr.log("planning", rr.TextLog(f"State: {state}, Turn Angle: {turn_angle:.1f}°"))
            
            # Log detections with both image and boxes
            rr.log(
                "detections/image",
                rr.Image(rgb).compress(jpeg_quality=85)
            )
            rr.log(
                "detections/boxes",
                rr.Boxes2D(
                    array=all_detections[:, :4],
                    array_format=rr.Box2DFormat.XYWH,
                    class_ids=all_detections[:, -1].astype(np.uint16),
                )
            )
            
            # Log boxes on depth map
            rr.log(
                "depth_map/boxes",
                rr.Boxes2D(
                    array=all_detections[:, :4],
                    array_format=rr.Box2DFormat.XYWH,
                    class_ids=all_detections[:, -1].astype(np.uint16),
                )
            )
            
            # Log distances and TTC for each detection
            info_text = ""
            for i, det in enumerate(all_detections):
                x, y, w, h = det[:4]
                class_id = int(det[-1])
                distance = planner.get_object_distance(det, depth_map)
                ttc = planner.estimate_ttc(
                    np.array([x + w/2, y + h/2]),
                    velocities.get(i, np.zeros(2)),
                    distance
                )
                
                # Convert class_id to name for better readability
                class_names = {
                    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
                    5: "bus", 7: "truck"
                }
                class_name = class_names.get(class_id, f"unknown_{class_id}")
                
                info_text += f"Box #{i} ({class_name}): dist={distance:.2f}m, TTC={ttc:.2f}s\n"
            rr.log("distances", rr.TextLog(info_text))
            
            # Visualize velocity vectors
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
        else:
            # If no detections, still show the image but clear boxes
            rr.log("detections/image", rr.Image(rgb).compress(jpeg_quality=85))
            rr.log("planning", rr.TextLog("State: FORWARD, Turn Angle: 0.0°"))
            rr.log("distances", rr.TextLog("No objects detected"))
        
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