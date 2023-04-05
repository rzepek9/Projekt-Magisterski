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

"""
Działanie pliku 
- flaga source sciezka do fold.txt
- uzyj flagi site oraz train aby wygenerowac csv dla foldu treningowego z bocznej perspektywy, wazne aby umiescic dobra sciezke 
- uyztj flagi site oraz test dla foldu testowego z bocznej perspektywy
- analogicnzie dla dolnej perspektywy

Wygenerowany csv posiada kp w stylu xyxyxy... oraz na koncu target w przypadku dolnej perspektywy, bez kp twarzy
Dla bocznej perspektywy kp w tym samym stylu tylko przed targetem dodane zostaly 4 katy (2 w lokciach , 2 w kolanach) KATY SA DO POPRAWY

"""

@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt",source='/home/s175668/raid/Praca-Magisterska/four_seconds/under',device='0', test = False, train= False, site = False, under = False):
    
    with open(source, 'r') as file:
        fold_lines = [line.strip() for line in file]
    
    device = select_device(opt.device) #select device

    model = attempt_load(poseweights, map_location=device)  #Load model
    _ = model.eval()
    
   
    for clip in fold_lines:

        clip_path = clip.replace("mov", "avi")
        file_name = clip_path.rsplit('/', 1)[-1]
        
        if site:
            calculate_angle = True
            clip_path = Path('/home/s175668/raid/Praca-Magisterska/shoots_only/site')
            if test:
                output_csv_dir = Path('/home/s175668/raid/Praca-Magisterska/Repozytorium/Projekt-Magisterski/workspace/test/fold0/site')
            if train:
                output_csv_dir = Path('/home/s175668/raid/Praca-Magisterska/Repozytorium/Projekt-Magisterski/workspace/train/fold0/site')
            
        if under:
            file_name = file_name.replace("side", "site")
            file_name = file_name.replace("site", "under")
            clip_path = Path('/home/s175668/raid/Praca-Magisterska/shoots_only/under')
            calculate_angle = False
            if test:
                output_csv_dir = Path('/home/s175668/raid/Praca-Magisterska/Repozytorium/Projekt-Magisterski/workspace/test/fold0/under')
            if train:
                output_csv_dir = Path('/home/s175668/raid/Praca-Magisterska/Repozytorium/Projekt-Magisterski/workspace/train/fold0/under')
        
        frame_features = []
        output_csv_dir = output_csv_dir / file_name.split('_')[0]
        target = int(clip[-5])
    
        clip_path = clip_path / file_name
        cap = cv2.VideoCapture(str(clip_path))

        frame_width = int(cap.get(3))  #get video frame width
        frame_height = int(cap.get(4)) #get video frame height
        frame_number = 1
        
        
        while(cap.isOpened): #loop until cap opened or video not complete
            created = False
            ret, frame = cap.read()  

            if ret: #if success is true, means frame exist
                print('frame_number,: ', frame_number)
                

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
                                            0.50,   # Conf. Threshold.
                                            0.65, # IoU Threshold.
                                            nc=model.yaml['nc'], # Number of classes.
                                            nkpt=model.yaml['nkpt'], # Number of keypoints.
                                            kpt_label=True)

                for pose in output_data:
                        for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])): #loop over poses for drawing on frame
                            if (site and pose[det_index, 21]/640 < 0.6) or created:
                                continue
                            else:
                                kpts = pose[det_index, 21:]
                                frame_features.append(frame_values(kpts, steps=3, target=target,calculate = calculate_angle))
                                created = True

                frame_number +=1
                
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

        frame_features = frame_features[:11]
        create_csv(frame_features, output_csv_dir, calculate_angle)
        assert len(frame_features) == 11 or file_name ==  '479_tomek_under_0.avi' or file_name == '338_gustaw_side_0.avi',(file_name, len(frame_features))



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='', help='fold.txt path') 
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)') 
    parser.add_argument('--test', action='store_true', default= False)
    parser.add_argument('--train', action='store_true', default= False)
    parser.add_argument('--site', action='store_true', default= False)
    parser.add_argument('--under', action='store_true', default= False)
    opt = parser.parse_args()
    return opt


def create_csv(values, output_dir, calculate_angle):
    if  not calculate_angle:
        header = ["LEFT_SHOULDER_X", "LEFT_SHOULDER_Y", "RIGH_SHOULDER_X", "RIGH_SHOULDER_Y", "LEFT_ELBOW_X", "LEFT_ELBOW_Y", "RIGHT_ELBOW_X", "RIGHT_ELBOW_Y", "LEFT_WRIST_X", "LEFT_WRIST_Y", "RIGHT_WRIST_X",
                "RIGHT_WRIST_Y", "LEFT_HIPS_X", "LEFT_HIPS_Y", "RIGHT_HIPS_X", "RIGHT_HIPS_Y", "LEFT_KNEE_X", "LEFT_KNEE_Y", "RIGHT_KNEE_X", "RIGHT_KNEE_Y",
                "LEFT_ANKLE_X","LEFT_ANKLE_Y", "RIGHT_ANKLE_X", "RIGHT_ANKLE_Y", "TARGET"]
    else:
        header = ["LEFT_SHOULDER_X", "LEFT_SHOULDER_Y", "RIGH_SHOULDER_X", "RIGH_SHOULDER_Y", "LEFT_ELBOW_X", "LEFT_ELBOW_Y", "RIGHT_ELBOW_X", "RIGHT_ELBOW_Y", "LEFT_WRIST_X", "LEFT_WRIST_Y", "RIGHT_WRIST_X",
                "RIGHT_WRIST_Y", "LEFT_HIPS_X", "LEFT_HIPS_Y", "RIGHT_HIPS_X", "RIGHT_HIPS_Y", "LEFT_KNEE_X", "LEFT_KNEE_Y", "RIGHT_KNEE_X", "RIGHT_KNEE_Y",
                "LEFT_ANKLE_X","LEFT_ANKLE_Y", "RIGHT_ANKLE_X", "RIGHT_ANKLE_Y", "RIGHT_ELBOW_ANGLE", "LEFT_ELBOW_ANGLE", "RIGHT_KNEE_ANGLE", "RIGHT_KNEE_ANGLE", "TARGET"]
    
    with open(f'{str(output_dir)}.csv', 'w') as f:
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
