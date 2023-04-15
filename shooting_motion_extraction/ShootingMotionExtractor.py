from pathlib import Path

import cv2
import torch
from numpy.typing import NDArray

from basketball_detection.BasketballDetector import BasketballDetector
from Yolov7_pose.pose_estimation.utils import preprocess_image
from Yolov7_pose.pose_estimation.YoloPose import YoloPose

from .utils import (
    get_distances,
    init_extraction_vars,
    is_ball_too_big,
    is_ball_above_knees,
)

OUTPUT_FILENAME = Path("extracted_shot.avi")
DIST_THRESHOLD = 100
FRAMES_THRESHOLD = 20


class ShootingMotionExtractor:
    """
    Class for extracting basketball shooting motion from a video
    """
    def __init__(
        self, yolo_bball_detector: BasketballDetector = None, yolo_pose: YoloPose = None
    ):
        self.bball_detector = (
            BasketballDetector() if not yolo_bball_detector else yolo_bball_detector
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_pose = YoloPose() if not yolo_pose else yolo_pose

    def get_detections(
        self, frame: NDArray, frame_width: int, bball_detection_conf: int = 0.5
    ) -> tuple:
        """
        Detects basketball objects and pose estimations
        Returns a tuple with both detections
        """
        image = preprocess_image(frame, frame_width, self.device)
        kpt_detections = self.yolo_pose.get_detections(image)
        basketball_detections = self.bball_detector.get_detections(
            frame, bball_detection_conf
        )
        return (basketball_detections, kpt_detections)

    def extract_shooting_motion_period(
        self, source: Path, bball_detection_conf: int = 0.5
    ) -> tuple:
        """
        Extracts the period of shooting motion from a video
        Returns a tuple with start and end time of the shooting period
        """
        cap = cv2.VideoCapture(str(source))
        frame_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if not cap.isOpened():
            print("Error while trying to read video. Check if source is valid")
            raise SystemExit()
        frames_since_ball, shooting_motion_start, shooting_motion_end = init_extraction_vars()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            basketball_detections, kpt_detections = self.get_detections(
                frame, frame_size[0], bball_detection_conf
            )
            if self.is_ball_in_hands(basketball_detections, kpt_detections):
                if not shooting_motion_start:
                    shooting_motion_start = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    print(f"Shooting motion started at {shooting_motion_start}")
                frames_since_ball = 0
            else:
                frames_since_ball += 1
                if frames_since_ball > FRAMES_THRESHOLD:
                    shooting_motion_end = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    print(f"Shooting motion ended at {shooting_motion_end}")
                    break
        if shooting_motion_start and not shooting_motion_end:
            shooting_motion_end = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        return (shooting_motion_start, shooting_motion_end)

    def extract_and_save_shooting_motion(
        self,
        source: Path,
        bball_detection_conf: int = 0.5,
        output_filename: Path = None,
    ) -> None | tuple:
        """
        Extracts the shooting motion for a video and allows for saving the extracted video
        and returning the shooting motion period
        """
        cap = cv2.VideoCapture(str(source))
        frame_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if not output_filename:
            output_filename = OUTPUT_FILENAME
            if isinstance(output_filename, Path):
                output_filename = str(output_filename)
        output_video = None
        if not cap.isOpened():
            print("Error while trying to read video. Check if source is valid")
            raise SystemExit()
        frames_since_ball, shooting_motion_start, shooting_motion_end = init_extraction_vars()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            basketball_detections, kpt_detections = self.get_detections(
                frame, frame_size[0], bball_detection_conf
            )
            if self.is_ball_in_hands(basketball_detections, kpt_detections):
                if not shooting_motion_start:
                    shooting_motion_start = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    print(f"Shooting motion started at {shooting_motion_start}")
                print("BALL IN HANDS")
                frames_since_ball = 0
                if not output_video:
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    if not output_filename:
                        output_filename = OUTPUT_FILENAME
                    if isinstance(output_filename, Path):
                        output_filename = str(output_filename)
                    output_video = cv2.VideoWriter(output_filename, fourcc, 20.0, frame_size)
            else:
                print("BALL NOT IN HANDS")
                frames_since_ball += 1
                if frames_since_ball > 200:
                    print(
                        f"{FRAMES_THRESHOLD} FRAMES WITHOUT THE BALL IN HANDS, STOPPING FILE WRITING"
                    )
                    shooting_motion_end = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    print(f"Shooting motion ended at {shooting_motion_end}")
                    break
            if output_video:
                output_video.write(frame)
        output_video.release()
        cap.release()
        cv2.destroyAllWindows()

    def is_ball_in_hands(self, basketball_detections: list, kpt_detections: list) -> bool:
        """
        Checks if the ball is in the hands of the player
        Returns boolean value of the check
        """
        if self.bball_detector.are_objects_detected(basketball_detections):
            boxes = basketball_detections[0].boxes.boxes.numpy()
            balls_detected = {
                i: boxes[i][:4] for i in range(len(boxes)) if boxes[i][-1] == 0
            }
            for key, ball in balls_detected.items():
                for pose in kpt_detections:  # detections per image
                    if self.yolo_pose.is_pose_detected(
                        kpt_detections
                    ):  # check if no pose
                        for det_index, (*xyxy, conf, cls) in enumerate(
                            reversed(pose[:, :6])
                        ):  # loop over poses for drawing on frame
                            kpts_without_head = pose[det_index, 21:]
                            if is_ball_above_knees(ball, kpts_without_head) and not is_ball_too_big(ball):
                                distances = get_distances(ball, kpts_without_head)
                                if any(
                                    dist <= DIST_THRESHOLD for dist in distances
                                ):
                                    return True
            return False