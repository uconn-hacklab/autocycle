#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path

import cv2
import rerun as rr
from ultralytics import YOLO

DESCRIPTION = """
# Autocycle with YOLOv8
This script demonstrates object detection and tracking using YOLOv8 and logs both raw and processed frames to Rerun.
""".strip()

FRAME_SKIP = 2  # Process every 2nd frame

def setup_logging() -> None:
    logger = logging.getLogger()
    rerun_handler = rr.LoggingHandler("logs")
    rerun_handler.setLevel(-1)
    logger.addHandler(rerun_handler)

def track_objects(video_path: str, max_frame_count: int | None) -> None:
    logging.info("Initializing YOLOv8 model...")
    model = YOLO("yolov8n.pt")
    logging.info("YOLOv8 model loaded.")

    logging.info("Loading input video: %s", video_path)
    cap = cv2.VideoCapture(video_path)

    frame_idx = 0

    while cap.isOpened():
        if max_frame_count is not None and frame_idx >= max_frame_count:
            break

        ret, bgr = cap.read()
        rr.set_time_sequence("frame", frame_idx)

        if not ret:
            logging.info("End of video")
            break

        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rr.log("video/raw_frame", rr.Image(rgb).compress(jpeg_quality=85))

        results = model.predict(rgb, imgsz=640, conf=0.7, device="0")
        output_frame = rgb.copy()

        if results:
            for result in results:
                for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                    x_min, y_min, x_max, y_max = map(int, box.tolist())
                    class_id = int(cls.item())
                    cv2.rectangle(output_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(
                        output_frame,
                        f"Class {class_id}",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
        else:
            logging.warning("No detections on frame %d", frame_idx)

        rr.log("video/detected_frame", rr.Image(output_frame).compress(jpeg_quality=85))
        logging.info("Processed frame %d", frame_idx)
        frame_idx += 1

    cap.release()

def main() -> None:
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel("DEBUG")

    parser = argparse.ArgumentParser(description="Object detection and tracking with YOLOv8.")
    parser.add_argument("--video-path", type=str, required=True, help="Full path to the video.")
    parser.add_argument("--max-frame", type=int, help="Maximum number of frames to process.")
    rr.script_add_args(parser)
    args = parser.parse_args()

    rr.script_setup(args, "autocycle_yolov8")
    setup_logging()

    rr.log("description", rr.TextDocument(DESCRIPTION, media_type=rr.MediaType.MARKDOWN), static=True)

    video_path = args.video_path
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video path '{video_path}' does not exist.")

    track_objects(video_path, max_frame_count=args.max_frame)
    rr.script_teardown(args)


if __name__ == "__main__":
    main()
