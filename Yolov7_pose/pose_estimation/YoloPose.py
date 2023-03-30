import os
import time
from pathlib import Path

import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt
from utils.plots import frame_values, plot_skeleton_without_head

from .utils import (create_csv, postprocess_image, preprocess_image,
                    write_video_results)

CURRENT_DIR = os.getcwd()
POSEWEIGHTS = "Yolov7_pose/yolov7-w6-pose.pt"
OUTPUT_VIDEO_DIR = Path("Yolov7_pose/output_videos")
OUTPUT_CSV_DIR = Path("Yolov7_pose/output_csv")


class YoloPose:
    def __init__(self, poseweights=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolov7_pose = (
            self.load_model(POSEWEIGHTS, self.device)
            if not poseweights
            else self.load_model(poseweights, self.device)
        )

    def load_model(self, poseweights, device):
        model = attempt_load(POSEWEIGHTS, device)
        return model.eval()

    def init_detection_vars(self):
        return 0, 0, [], [], [], []

    def detect_on_video(
        self,
        source,
        show_video=True,
        save_video=False,
        write_cords_to_csv=False,
        output_filename=None,
    ):
        (
            frame_count,
            total_fps,
            time_list,
            fps_list,
            frame_cords,
            results_img,
        ) = self.init_detection_vars()
        if not output_filename:
            output_filename = f"{source.split('/')[-1].split('.')[0]}"
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error while trying to read video. Check if source is valid")
            raise SystemExit()
        frame_width = int(cap.get(3))  # get video frame width
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            image = preprocess_image(frame, frame_width, self.device)
            start_time = time.time()  # start time for fps calculation
            detections = self.get_detections(image)
            frame_cords, image_post = self.plot_detections_and_get_results(
                image, detections, frame_cords, frame_count
            )
            results_img.append(image_post)
            end_time = time.time()  # Calculatio for FPS
            fps = 1 / (end_time - start_time)
            total_fps += fps
            frame_count += 1
            fps_list.append(total_fps)  # append FPS in list
            time_list.append(end_time - start_time)  # append time in list
            if show_video:
                cv2.imshow("Pose Estimation", image_post)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        if save_video:
            write_video_results(results_img, OUTPUT_VIDEO_DIR, output_filename)
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")
        if write_cords_to_csv:
            create_csv(frame_cords, OUTPUT_CSV_DIR, output_filename)
        cap.release()
        cv2.destroyAllWindows()

    def is_pose_detected(self, detections):
        return len(detections) != 0

    def plot_detections_and_get_results(
        self, image, detections, frame_cords=[], frame_count=0
    ):
        image_post = postprocess_image(image)
        for pose in detections:  # detections per image
            if self.is_pose_detected(detections):  # check if no pose
                for c in pose[:, 5].unique():  # Print results
                    n = (pose[:, 5] == c).sum()  # detections per class
                    print("No of Objects in Current Frame : {}".format(n))
                for det_index, (*xyxy, conf, cls) in enumerate(
                    reversed(pose[:, :6])
                ):  # loop over poses for drawing on frame
                    c = int(cls)  # integer class
                    kpts_without_head = pose[det_index, 21:]
                    # tworzy polaczenia dla nog i rak tylko
                    plot_skeleton_without_head(
                        image_post,
                        kpts_without_head,
                        steps=3,
                        orig_shape=image_post.shape[:2],
                    )
                    # wyciagamy wspolrzedne z klatki
                    if frame_count % 2 == 0 and frame_count < 40:
                        frame_cords.append(
                            frame_values(frame_count, kpts_without_head, 3)
                        )
                    frame_count += 1
        return frame_cords, image_post

    def get_detections(self, image):
        with torch.no_grad():  # get predictions
            output_data, _ = self.yolov7_pose(image)
            return non_max_suppression_kpt(
                output_data,  # Apply non max suppression
                0.25,  # Conf. Threshold.
                0.65,  # IoU Threshold.
                nc=self.yolov7_pose.yaml["nc"],  # Number of classes.
                nkpt=self.yolov7_pose.yaml["nkpt"],  # Number of keypoints.
                kpt_label=True,
            )
