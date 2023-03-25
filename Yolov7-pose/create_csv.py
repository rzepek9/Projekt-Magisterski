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
from utils.general import non_max_suppression_kpt,strip_optimizer,xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt, plot_skeleton_without_head, frame_values

import traceback
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
def run(poseweights="yolov7-w6-pose.pt",fold_source='/home/s175668/raid/Praca-Magisterska/four_seconds/under',device='0'):
    
    with open(fold_source, 'r') as file:
        fold_lines = [line.strip() for line in file]
    
    device = select_device(opt.device) #select device

    model = attempt_load(poseweights, map_location=device)  #Load model
    _ = model.eval()
   
    for clip in fold_lines:
        
        if "side" in clip


        if 'site' not in source:
            calculate_angle = False
            output_csv_dir = output_csv_dir / 'under'
        else:
            output_csv_dir = output_csv_dir / 'site'
            calculate_angle = True

        frame_cords = []
        target = int(source[-5])
        file_name = source.split('_', 1)[0]
        cap = cv2.VideoCapture(f'{source_dir}/{source}')

        if (cap.isOpened() == False):   #check if videocapture not opened
            print('Error while trying to read video. Please check path again')
            raise SystemExit()
        

        else:
            frame_width = int(cap.get(3))  #get video frame width
            frame_height = int(cap.get(4)) #get video frame height

            while(cap.isOpened): #loop until cap opened or video not complete
            

                ret, frame = cap.read()  
                
                if ret: #if success is true, means frame exist

                    orig_image = frame #store frame
                    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) #convert frame to RGB
                    image = letterbox(image, (frame_width), stride=64, auto=True)[0]

                    image = transforms.ToTensor()(image)
                    image = torch.tensor(np.array([image.numpy()]))
                
                    image = image.to(device)  #convert image data to device
                    image = image.float() #convert image to float precision (cpu)
                   
                
                    with torch.no_grad():  #get predictions
                        output_data, _ = model(image)


                    output_data = non_max_suppression_kpt(output_data,   #Apply non max suppression
                                                0.75,   # Conf. Threshold.
                                                0.65, # IoU Threshold.
                                                nc=model.yaml['nc'], # Number of classes.
                                                nkpt=model.yaml['nkpt'], # Number of keypoints.
                                                kpt_label=True)


                    for pose in output_data:  # detections per image
                        if len(output_data):  #check if no pose
                            for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])): #loop over poses for drawing on frame
            
                            
                                kpts = pose[det_index, 21:]
                                frame_cords.append(frame_values(kpts, steps=3, target=target,calculate = calculate_angle))
    
                
                else:
                    break

            cap.release()
            cv2.destroyAllWindows()

            # tworze csv
            create_csv(frame_cords, output_csv_dir, file_name)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='', help='video/0 for webcam') #fold_source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   #device arugments
    opt = parser.parse_args()
    return opt


def create_csv(values, output_dir, file_name, under= True):
    if under:
        header = ["LEFT_SHOULDER_X", "LEFT_SHOULDER_Y", "RIGH_SHOULDER_X", "RIGH_SHOULDER_Y", "LEFT_ELBOW_X", "LEFT_ELBOW_Y", "RIGHT_ELBOW_X", "RIGHT_ELBOW_Y", "LEFT_WRIST_X", "LEFT_WRIST_Y", "RIGHT_WRIST_X",
                "RIGHT_WRIST_Y", "LEFT_HIPS_X", "LEFT_HIPS_Y", "RIGHT_HIPS_X", "RIGHT_HIPS_Y", "LEFT_KNEE_X", "LEFT_KNEE_Y", "RIGHT_KNEE_X", "RIGHT_KNEE_Y",
                "LEFT_ANKLE_X","LEFT_ANKLE_Y", "RIGHT_ANKLE_X", "RIGHT_ANKLE_Y", "TARGET"]
        csv_dir = output_dir / 'under' / f'{file_name}.csv'
    else:
        header = ["LEFT_SHOULDER_X", "LEFT_SHOULDER_Y", "RIGH_SHOULDER_X", "RIGH_SHOULDER_Y", "LEFT_ELBOW_X", "LEFT_ELBOW_Y", "RIGHT_ELBOW_X", "RIGHT_ELBOW_Y", "LEFT_WRIST_X", "LEFT_WRIST_Y", "RIGHT_WRIST_X",
                "RIGHT_WRIST_Y", "LEFT_HIPS_X", "LEFT_HIPS_Y", "RIGHT_HIPS_X", "RIGHT_HIPS_Y", "LEFT_KNEE_X", "LEFT_KNEE_Y", "RIGHT_KNEE_X", "RIGHT_KNEE_Y",
                "LEFT_ANKLE_X","LEFT_ANKLE_Y", "RIGHT_ANKLE_X", "RIGHT_ANKLE_Y", "RIGHT_ELBOW_ANGLE", "LEFT_ELBOW_ANGLE", "RIGHT_KNEE_ANGLE", "RIGHT_KNEE_ANGLE", "TARGET"]
        csv_dir = output_dir / 'site' / f'{file_name}.csv'    
    
    if not csv_dir.exists():
        with open(str(csv_dir), 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(values)
    
    

#main function
def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device,opt.poseweights)
    main(opt)
