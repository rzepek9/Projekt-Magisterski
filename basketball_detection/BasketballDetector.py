from pathlib import Path

import cv2
from numpy.typing import NDArray
from ultralytics import YOLO
import supervision as sv

YOLO_BBALL_CHECKPOINT = Path("basketball_detection/yolov8_bball.pt")
OUTPUT_FILENAME = Path("basketball_detection/basketball_detection.avi")


class BasketballDetector:
    """
    Class for detecting basketball objects (basketballs and baskets)
    """

    def __init__(self, yolo_weights: Path = None) -> None:
        self.yolov8 = (
            YOLO(YOLO_BBALL_CHECKPOINT) if not yolo_weights else YOLO(yolo_weights)
        )
        self.id2label = {0: "ball", 1: "basket"}
        self.box_annotator = sv.BoxAnnotator()

    def get_detections(self, frame: NDArray, confidence_threshold: int = 0.5) -> sv.Detections:
        """
        Makes predictions on an image with a given confidence
        Returns supervision Detections
        """
        yolo_detections = self.yolov8.predict(source=[frame], conf=confidence_threshold, verbose=False)
        return sv.Detections.from_yolov8(yolo_detections[0])

    def plot_detections(self, frame: NDArray, detections: sv.Detections) -> None:
        """
        Plots detected basketball objects on a frame based
        on supervision Detections
        """
        labels = [
            f"{self.id2label[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(), detections=detections, labels=labels
        )
        return annotated_frame

    def detect_on_video(
        self,
        source: Path,
        confidence_threshold: int = 0.5,
        show_video: bool = True,
        save_video: bool = False,
        output_filename: Path = None,
    ) -> None:
        """
        Makes detections on a video with a given confidence
        Allows for:
        displaying the video
        saving the video with detections to a default/selected path
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error while trying to read video. Check if source is valid")
            raise SystemExit()
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            if not output_filename:
                output_filename = OUTPUT_FILENAME
            frame_size = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            output_video = cv2.VideoWriter(
                str(output_filename), fourcc, 20.0, frame_size
            )
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            frame = resize_frame(frame)
            detections = self.get_detections(frame, confidence_threshold)
            annotated_frame = self.plot_detections(frame, detections)
            if save_video:
                output_video.write(annotated_frame)
            if show_video:
                cv2.imshow("Basketball Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        if save_video:
            output_video.release()
        cap.release()
        cv2.destroyAllWindows()


def resize_frame(frame, target_width=640, target_height=480):
    """
    Resize the frame to the target width and height while maintaining the aspect ratio.
    """
    aspect_ratio = frame.shape[1] / frame.shape[0]
    target_width = min(target_width, int(target_height * aspect_ratio))
    target_height = int(target_width / aspect_ratio)
    return cv2.resize(frame, (target_width, target_height))
