import csv
import os
from subprocess import PIPE, Popen

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from Yolov7_pose.utils.datasets import letterbox


def preprocess_image(frame, frame_width, device):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert frame to RGB
    image = letterbox(image, (frame_width), stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    image = image.to(device)  # convert image data to device
    image = image.float()  # convert image to float precision (cpu)
    return image


def postprocess_image(image):
    im0 = image[0].permute(1, 2, 0) * 255
    im0 = im0.cpu().numpy().astype(np.uint8)
    return cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)


def create_csv(values, output_dir, output_filename):
    header = [
        "Frame nr",
        "LEFT_SHOULDER",
        "RIGH_SHOULDER",
        "LEFT_ELBOW",
        "RIGHT_ELBOW",
        "LEFT_WRIST",
        "RIGHT_WRIST",
        "LEFT_HIPS",
        "RIGHT_HIPS",
        "LEFT_KNEE",
        "RIGHT_KNEE",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
        "LEFT_ELBOW_ANGLE",
        "RIGHT_ELBOW_ANGLE",
        "LEFT_KNEE_ANGLE",
        "RIGHT_KNEE_ANGLE",
    ]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    csv_dir = output_dir / f"{output_filename}.csv"
    if not csv_dir.exists():
        with open(str(csv_dir), "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(values)


def write_video_results(results_img, output_dir, output_filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    p = Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-r",
            "24",
            "-i",
            "-",
            "-vcodec",
            "mpeg4",
            "-qscale",
            "5",
            "-r",
            "10",
            f"{output_dir}/{output_filename}.avi",
        ],
        stdin=PIPE,
    )
    for img in results_img:
        im = Image.fromarray(img[:, :, ::-1])
        im.save(p.stdin, "JPEG")
    p.stdin.close()
    p.wait()
