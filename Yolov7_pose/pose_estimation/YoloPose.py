from pathlib import Path

import cv2
import torch
from numpy.typing import NDArray

from ..models.experimental import attempt_load
from ..utils.general import non_max_suppression_kpt
from ..utils.plots import frame_values, plot_skeleton_without_head
from .utils import (init_detection_vars, postprocess_image, preprocess_image,
                    write_video_results)

POSEWEIGHTS = Path("Yolov7_pose/yolov7-w6-pose.pt")
OUTPUT_VIDEO_DIR = Path("Yolov7_pose/output_videos")


class YoloPose:
    """
    Class for pose estimation with Yolov7
    One notice:
    Due to import problems conflicts with the original Yolov7_pose repo
    to use this class you need to have a Yolov7_pose path set in your pythonpath
    or append it to your sys.path
    """

    def __init__(self, poseweights: Path = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolov7_pose = (
            self.load_model(POSEWEIGHTS)
            if not poseweights
            else self.load_model(poseweights)
        )

    def load_model(self, poseweights: Path) -> torch.nn.Sequential:
        """
        Loads the model from weigths path and sets it to evaluation mode
        """
        model = (
            attempt_load(poseweights, self.device)
            if poseweights
            else attempt_load(POSEWEIGHTS, self.device)
        )
        return model.eval()

    def detect_on_video(
        self,
        source: Path,
        show_video: bool = True,
        save_video: bool = False,
        output_filename: Path = None,
    ) -> None:
        """
        Makes detections on a video
        Allows for:
        displaying the video
        saving the video with detections
        """
        frame_count, frame_cords, results_img = init_detection_vars()
        if save_video and not output_filename:
            if isinstance(output_filename, Path):
                output_filename = str(output_filename)
            output_filename = f"{source.split('/')[-1].split('.')[0]}"
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error while trying to read video. Check if source is valid")
            raise SystemExit()
        frame_width = int(cap.get(3))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            image = preprocess_image(frame, frame_width, self.device)
            detections = self.get_detections(image)
            frame_cords, image_post = self.plot_detections_and_get_results(
                image, detections, frame_cords, frame_count
            )
            results_img.append(image_post)
            frame_count += 1
            if show_video:
                cv2.imshow("Pose Estimation", image_post)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        if save_video:
            write_video_results(results_img, OUTPUT_VIDEO_DIR, output_filename)
        cap.release()
        cv2.destroyAllWindows()

    def is_pose_detected(self, detections: list) -> bool:
        """
        Checks if objects are detected and returns according bool value
        """
        return len(detections) != 0

    def plot_detections_and_get_results(
        self,
        image: NDArray,
        detections: list,
        frame_cords: list = [],
        frame_count: int = 0,
    ) -> tuple:
        """
        Plots detections on image
        based on a list of detections
        Returns frame coordinates and the result image
        """
        image_post = postprocess_image(image)
        for pose in detections:
            if self.is_pose_detected(detections):
                for c in pose[:, 5].unique():
                    n = (pose[:, 5] == c).sum()
                    print("No of Objects in Current Frame : {}".format(n))
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:, :6])):
                    c = int(cls)
                    kpts_without_head = pose[det_index, 21:]
                    plot_skeleton_without_head(
                        image_post,
                        kpts_without_head,
                        steps=3,
                        orig_shape=image_post.shape[:2],
                    )
                    if frame_count % 2 == 0 and frame_count < 40:
                        frame_cords.append(
                            frame_values(frame_count, kpts_without_head, 3)
                        )
                    frame_count += 1
        return frame_cords, image_post

    def get_detections(self, image: NDArray) -> torch.tensor:
        """
        Makes detections on an image
        Returns detections in form of a tensor
        """
        with torch.no_grad():
            output_data, _ = self.yolov7_pose(image)
            return non_max_suppression_kpt(
                output_data,
                0.25,
                0.65,
                nc=self.yolov7_pose.yaml["nc"],
                nkpt=self.yolov7_pose.yaml["nkpt"],
                kpt_label=True,
            )
