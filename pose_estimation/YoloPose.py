from pathlib import Path
import cv2
from ultralytics import YOLO
from numpy.typing import NDArray

YOLO_POSE_CHECKPOINT = "yolov8n-pose.pt"
OUTPUT_VIDEO_DIR = Path("Yolov7_pose/output_videos")


class YoloPose:
    """
    Class for pose estimation with Yolov8 using Ultralytics library
    """

    def __init__(self, yolo_weights: str = YOLO_POSE_CHECKPOINT) -> None:
        self.yolov8 = YOLO(yolo_weights)

    def get_keypoints_and_bboxes(self, results):
        """
        Makes predictions on an image with a given confidence
        Returns Ultralytics detections
        """
        return (
            [r.keypoints.xy.cpu().numpy() for r in results],
            [r.boxes.xyxy.cpu().numpy() for r in results]
        )

    def get_detections(self, frame: NDArray, confidence_threshold: int = 0.5):
        """
        Makes predictions on an image with a given confidence
        Returns Ultralytics detections
        """
        return self.yolov8.predict(source=[frame], conf=confidence_threshold, verbose=False)

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
                output_filename = OUTPUT_VIDEO_DIR / "output_video.avi"
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
            results = self.get_detections(frame, confidence_threshold)
            annotated_frame = results[0].plot()
            if save_video:
                output_video.write(annotated_frame)
            if show_video:
                cv2.imshow("Pose Estimation", annotated_frame)
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
