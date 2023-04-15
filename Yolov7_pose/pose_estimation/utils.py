import os
from pathlib import Path
from subprocess import PIPE, Popen

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torchvision import transforms

from Yolov7_pose.utils.datasets import letterbox


def init_detection_vars() -> tuple:
    """
    Initializes detection variables for the YoloPose model
    """
    return 0, [], []


def preprocess_image(frame: NDArray, frame_width: int, device: torch.device):
    """
    Preprocesses input image
    """
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert frame to RGB
    image = letterbox(image, (frame_width), stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    image = image.to(device)  # convert image data to device
    image = image.float()  # convert image to float precision (cpu)
    return image


def postprocess_image(image: NDArray) -> cv2.cvtColor:
    """
    Postprocesses the image
    """
    im0 = image[0].permute(1, 2, 0) * 255
    im0 = im0.cpu().numpy().astype(np.uint8)
    return cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)


def write_video_results(
    results_img: NDArray, output_dir: Path, output_filename: Path
) -> None:
    """ "
    Writes the output video
    """
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
