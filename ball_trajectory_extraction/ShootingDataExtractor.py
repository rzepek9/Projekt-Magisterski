import cv2
import numpy as np
import supervision as sv

from basketball_detection import BasketballDetector, DetrForBasketballDetection
from pose_estimation import YoloPose


class ShootingDataExtractor:
    def __init__(
        self, pose_estimator: YoloPose, basketball_detector: BasketballDetector | DetrForBasketballDetection=None
    ) -> None:
        self.pose_estimator = pose_estimator
        self.basketball_detector = basketball_detector

    def collect_data_from_video(
        self, video_path, pose_confidence=0.35, ball_confidence=0.5, iou_threshold=0.8, perspective="under"
    ):
        cap = cv2.VideoCapture(video_path)
        collected_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get keypoints from YoloPose
            results = self.pose_estimator.get_detections(frame, pose_confidence)
            try:
                keypoints_all, bboxes_all = self.pose_estimator.get_keypoints_and_bboxes(results)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                keypoints_all, bboxes_all = [], []
            if len(keypoints_all) != 0 and len(bboxes_all) != 0:
                keypoints_shooting_person, bbox_shooting_person = self.determine_shooting_person(
                    keypoints_all, bboxes_all, perspective
                )
            else:
                keypoints_shooting_person, bbox_shooting_person = None, None

            balls = None
            baskets = None
            # Get ball and basket detections only for side perspective
            if perspective == "side":
                detections = (
                    self.basketball_detector.get_detections(frame, ball_confidence, iou_threshold)
                    if isinstance(self.basketball_detector, DetrForBasketballDetection)
                    else self.basketball_detector.get_detections(frame, ball_confidence)
                )

                ball_detections = self.filter_ball_detections(detections) if detections  is not None else None
                basket_detections = self.filter_basket_detections(detections) if detections  is not None else None

                balls = ball_detections.xyxy if ball_detections is not None else None
                baskets = basket_detections.xyxy if basket_detections is not None else None

            # Store the data for this frame
            frame_data = {
                "keypoints": keypoints_shooting_person,
                "person_bbox": bbox_shooting_person,
                "balls": balls,
                "baskets": baskets,
            }
            collected_data.append(frame_data)

        cap.release()
        return collected_data

    def determine_shooting_person(self, keypoints, bboxes, perspective="under"):
        if len(keypoints) == 0 or len(bboxes) == 0:
            return None, None

        if perspective == "under":
            return keypoints[0], bboxes[0]
        elif perspective == "side":
            # Filter out empty keypoints and ensure the first keypoint exists
            valid_indices = [i for i in range(len(keypoints)) if len(keypoints[i]) > 0 and len(keypoints[i][0]) > 0]

            # If there are no valid keypoints, return None
            if not valid_indices:
                return None, None

            # Sort by the rightmost x-coordinate of the bounding box to take the person most to the right
            final_indice = None
            if len(bboxes[0]) == 0:
                bboxes = None
            if len(keypoints[0]) == 0:
                keypoints = None
            elif len(bboxes[0]) > 0 and len(keypoints[0]) > 0:
                for i in range(len(bboxes)):
                    if final_indice is None:
                        final_indice = i
                    elif bboxes[i][2] > bboxes[final_indice][2]:
                        final_indice = i
                return keypoints[i], bboxes[i]
            else:
                return None, None

    def filter_ball_detections(self, detections):
        # Get the indices of detections with the label "ball"
        ball_indices = [
            i for i in range(len(detections.class_id))
            if self.basketball_detector.id2label[detections.class_id[i]] == "ball"
        ]

        # Check if ball_indices is empty
        if not ball_indices:
            return None

        # Create a new sv.Detections object with the filtered data
        filtered_detections = sv.Detections(
            xyxy=detections.xyxy[ball_indices],
            class_id=detections.class_id[ball_indices],
            confidence=detections.confidence[ball_indices],
        )
        return filtered_detections

    def filter_basket_detections(self, detections):
        # Get the indices of detections with the label "ball"
        basket_indices = [
            i for i in range(len(detections.class_id))
            if self.basketball_detector.id2label[detections.class_id[i]] == "basket"
        ]

        # Check if ball_indices is empty
        if not basket_indices:
            return None

        # Create a new sv.Detections object with the filtered data
        filtered_detections = sv.Detections(
            xyxy=detections.xyxy[basket_indices],
            class_id=detections.class_id[basket_indices],
            confidence=detections.confidence[basket_indices],
        )

        return filtered_detections
