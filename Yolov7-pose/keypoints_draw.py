import cv2
import time
import torch
import argparse
import csv
import os
from pathlib import Path

from PIL import Image
from subprocess import Popen, PIPE

import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt,strip_optimizer
from utils.plots import plot_skeleton_without_head, plot_vectors


"""
Działanie pliku 
-flaga source jest teraz sciezka do plikow
- przejscie przez wszystkie pliki i wyciagniecie co 2 klatki do 40 klatki, 
- wyciagniecie keypointow

keypointy znajduja sie w posie ktora przechowuje na odpowiednym miejscu osobe zdetektowana i jej keypointy
Generalna posatc listy keypointow to pierwsze 4 indeksy to box detekcji osoby 5 indeks to jego conf i 6 nie wiem co oznacza,
7 index to juz x cord pierwszej czesci ciala, 8 y cord tej czesci, 9 confidence tej czesci i tak leci nastepna czesc ciala (X,Y,conf)
kolejnosc jest nastepujaca
KEYPOINTS = ["NOSE", "LEFT_EYE", "RIGH_EYE", "LEFT_EAR", "RIGH_EYE", "LEFT_SHOULDER", "RIGH_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
             "LEFT_HIPS", "RIGHT_HIPS", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"]

obraz tego pokaze ci troche linijka 132 i 133 oraz funkcja plot_one_box_kpt
"""

@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt", source='/home/s175668/raid/Praca-Magisterska/four_seconds/under', device='0',
        test=False, train=False, site=False, under=False):

    source_list = Path(source).glob('*.avi')
    device = select_device(opt.device)  # select device
    model = attempt_load(poseweights, map_location=device).eval()  # Load model

    source_list = ['/home/s175668/raid/Praca-Magisterska/version_01/168_tomek_under_1.avi',
                   '/home/s175668/raid/Praca-Magisterska/version_01/402_tomek_side_1.avi',
                   '/home/s175668/raid/Praca-Magisterska/version_01/410_tomek_side_0.avi',
                   '/home/s175668/raid/Praca-Magisterska/version_01/411_tomek_under_0.avi',
                    '/home/s175668/raid/Praca-Magisterska/version_01/844_krystian_side_celny.avi',
                    '/home/s175668/raid/Praca-Magisterska/version_01/833_krystian_side_niecelny.avi',
                    '/home/s175668/raid/Praca-Magisterska/version_01/820_krystian_under_niecelny.avi',
                    '/home/s175668/raid/Praca-Magisterska/version_01/824_krystian_under_celny.avi',
                    '/home/s175668/raid/Praca-Magisterska/version_01/441_gustaw_under_0.avi',
                    '/home/s175668/raid/Praca-Magisterska/version_01/443_gustaw_side_0.avi',
                    '/home/s175668/raid/Praca-Magisterska/version_01/436_gustaw_side_1.avi',
                    '/home/s175668/raid/Praca-Magisterska/version_01/429_gustaw_under_1.avi',
                    '/home/s175668/raid/Praca-Magisterska/version_01/107_kuba_site_1.avi',
                    '/home/s175668/raid/Praca-Magisterska/version_01/109_kuba_site_0.avi',
                    '/home/s175668/raid/Praca-Magisterska/version_01/795_kuba_under_niecelny.avi',
                    '/home/s175668/raid/Praca-Magisterska/version_01/642_kuba_under_celny.avi']

    for clip in source_list:

        file_name = str(clip).rsplit('/', 1)[-1][:-4]
        file_name = file_name.replace('site', 'side')
        side = bool('side' in file_name)

        label = file_name.rsplit('_', 1)[-1]

        if label == 'celny':
            file_name = file_name.replace("celny", "1")
        elif label == 'niecelny':
            file_name = file_name.replace("niecelny", "0")
        

        cap = cv2.VideoCapture(str(clip))
        frame_number =0

        while (cap.isOpened):  # loop until cap opened or video not complete
            created = False
            ret, frame = cap.read()
            frame_width = int(cap.get(3))
            frame_hight = int(cap.get(4))

            if ret:  # if success is true, means frame exist

                orig_image = frame  # store frame
                
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) # convert frame to RGB
                image = letterbox(image, (frame_width), stride=64, auto=True)[0]

                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))

                image = image.to(device).float()  # convert image data to device

                with torch.no_grad():  # get predictions
                    output_data, _ = model(image)

                output_data = non_max_suppression_kpt(output_data,  # Apply non max suppression
                                                      0.8, # Conf. Threshold.
                                                      0.85,  # IoU Threshold.
                                                      nc=model.yaml['nc'],  # Number of classes.
                                                      nkpt=model.yaml['nkpt'], # Number of keypoints.
                                                      kpt_label=True)


                im0 = image[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
                im0 = im0.cpu().numpy().astype(np.uint8)
                
                im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) #reshape image format to (BGR)

                for pose in output_data:
                    for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:, :6])):
                        # Na pojedynczych filmikach wykrywa osobe pod koszem co wyplywalo na wynik csv
                        if (side and pose[det_index, 21]/640 < 0.35) or created:
                            continue
                        else:
                            nose = pose[det_index, 6:8]
                            kpts = pose[det_index, 21:]

                            plot_skeleton_without_head(im0, kpts, 3)
                            # plot_vectors(im0, kpts, 3, nose)

                            output_path = Path('/home/s175668/raid/Praca-Magisterska/visualization/keypoints_drawed_th_90') / f"{file_name}_{frame_number}.png"
                            cv2.imwrite(str(output_path), im0)
                            created = True

                frame_number += 1
            else:
                break
            

        cap.release()
        cv2.destroyAllWindows()



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='football1.mp4', help='video/0 for webcam') #video source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   #device arugments
    opt = parser.parse_args()
    return opt
    

#main function
def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device,opt.poseweights)
    main(opt)
