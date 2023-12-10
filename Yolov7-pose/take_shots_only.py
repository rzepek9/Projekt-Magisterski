import cv2
import time
import torch
import argparse
import csv
import os
from pathlib import Path

from PIL import Image
from subprocess import Popen, PIPE
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt,strip_optimizer, xyxy2xywh
from utils.plots import plot_skeleton_without_head, frame_values
from pathlib import Path


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
def run(poseweights="yolov7-w6-pose.pt",source='/home/s175668/raid/Praca-Magisterska/dataset/probka',device='0',view_img=False,
        save_conf=False,line_thickness = 1,hide_labels=False, hide_conf=True):
    
    output_dir = Path('/home/s175668/raid/Praca-Magisterska/shoots_only/site')

    source_dir = Path(source)
    source_list = source_dir.glob('*.avi')
    source_list = ['/home/s175668/raid/Praca-Magisterska/dataset/raw/gustaw_under_rename/4213_kuba_under_1.avi']

    device = select_device(opt.device) #select device

    model = attempt_load(poseweights, map_location=device)  #Load model
    _ = model.eval()
    bad_files = []
    print(source_dir)
    to_less_frames = []
    for source in source_list:
        frame_count = 0  #count no of frames

        results_img = []
        reference_frame = 0
        foot = 960
       


        file_name = str(source).rsplit('/', 1)[-1][:-4]
        print(source)
        
        # if (Path('/home/s175668/raid/Praca-Magisterska/dataset/preprocessed/gustaw_side') / f'{file_name}.avi').exists():
        #     print('exist')
        #     continue

        # site = bool("under" not in file_name)
        cap = cv2.VideoCapture(str(source))


        if (cap.isOpened() == False):   #check if videocapture not opened
            bad_files.append([file_name, 0, 0])
            print('Error while trying to read video. Please check path again')
            continue
        

        else:
            frame_width = int(cap.get(3))  #get video frame width
            frame_height = int(cap.get(4)) #get video frame height
            
            size = (frame_width, frame_height)


            while(cap.isOpened): #loop until cap opened or video not complete
            

                ret, frame = cap.read()
                results_img.append(frame)
                
                
                if ret: #if success is true, means frame exist
                    created = False
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


                    im0 = image[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
                    im0 = im0.cpu().numpy().astype(np.uint8)
                    
                    im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) #reshape image format to (BGR)
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                    for pose in output_data:  # detections per image
                        if len(output_data):  #check if no pose
                            for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])): #loop over poses for drawing on frame
                                if (pose[det_index, 21] < 500 ) or created or (pose[det_index, 37] < 180):
                                    continue
                                else:
                                    created = True
                                    wrist_kpt = pose[det_index, 37]
                                    print(wrist_kpt)
                                    print(frame_count)
                                    
                                    if  wrist_kpt < foot and wrist_kpt > 100:
                                        # cv2.imwrite
                                        foot = wrist_kpt
                                        reference_frame = frame_count
                                        
                                        
                        else:
                            print('bad frame')
                    frame_count +=1
                
                else:
                    break

            cap.release()
            cv2.destroyAllWindows()



        
        
            
    
        # p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-r', '24', '-i', '-', '-vcodec', 'mpeg4', '-qscale', '5', '-r', '10', f'/home/s175668/raid/Praca-Magisterska/output_maj/{file_name}.avi'], stdin=PIPE)
       
        # if reference_frame > len(results_img)-9:
        #     bad_files.append([file_name, reference_frame, frame_count])
        #     reference_frame = reference_frame - (reference_frame - (len(results_img)-9))
        
        # if reference_frame - 22 < 0:
        #     bad_files.append([file_name, reference_frame, frame_count])
        #     reference_frame = reference_frame + (22 - reference_frame)
            
        # try:
        #     for img in results_img[reference_frame-22:reference_frame+9]:
        #         im = Image.fromarray(img[:,:,::-1])
        #         im.save(p.stdin, 'JPEG')
        #     p.stdin.close()
        #     p.wait()
        # except:
        #     print(file_name)
        
        out = cv2.VideoWriter(f'/home/s175668/raid/Praca-Magisterska/dataset/preprocessed/gustaw_under/{file_name}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 20, size)
        
        # start = reference_frame - (11)
        start = 33 - (11)
        # stop = reference_frame + 7
        stop = 33 + 7
        print("IMG LEN: ", len(results_img[start:stop]))
        print(start)
        print(stop)
        if start < 0:
            to_less_frames.append([file_name, start, reference_frame])
            stop += abs(start)
            start = 0
            
        for im in results_img[start:stop]:
            out.write(im)
            
        
        
    # try:
    #     with open('/home/s175668/raid/Praca-Magisterska/output_maj/prob_badfiles.txt', 'w', newline='') as f:
    #         writer = csv.writer(f)
    #         for element in bad_files:
    #             print(element)
    #             writer.writerow(element)
    # except:
    #     print(bad_files)
    #     print("Problem with csv code")
        
    # less_frames = pd.DataFrame(to_less_frames, columns=['name', 'start', 'reference'])
    # less_frames.to_csv('/home/s175668/raid/Praca-Magisterska/dataset/gustaw_side.csv', index=False)




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='/home/s175668/raid/Praca-Magisterska/dataset/probka', help='video/0 for webcam') #video source
    parser.add_argument('--device', type=str, default='0', help='cpu/0,1,2,3(gpu)')   #device arugments
    opt = parser.parse_args()
    return opt
    

#main function
def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device,opt.poseweights)
    main(opt)
