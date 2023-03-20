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
def run(poseweights="yolov7-w6-pose.pt",source="football1.mp4",device='cpu',view_img=False,
        save_conf=False,line_thickness = 1,hide_labels=False, hide_conf=True):

    output_img_dir = Path('/home/s175668/raid/Praca-Magisterska/Repozytorium/Projekt-Magisterski/Yolov7-pose/output')
    output_csv_dir = Path('/home/s175668/raid/Praca-Magisterska/Repozytorium/Projekt-Magisterski/Yolov7-pose/csv_files')


    source_dir = source
    source_list = os.listdir(source_dir)

    file_name = source.rsplit('/', 1)[-1][:-4]
    
    device = select_device(opt.device) #select device

    model = attempt_load(poseweights, map_location=device)  #Load model
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
   
    for source in source_list:
        frame_count = 0  #count no of frames
        total_fps = 0  #count total fps
        time_list = []   #list to store time
        fps_list = []    #list to store fps
        frame_cords = []
        results_img = []

        file_name = source.rsplit('/', 1)[-1][:-4]

        cap = cv2.VideoCapture(f'{source_dir}/{source}')

        if (cap.isOpened() == False):   #check if videocapture not opened
            print('Error while trying to read video. Please check path again')
            raise SystemExit()
        

        else:
            frame_width = int(cap.get(3))  #get video frame width
            frame_height = int(cap.get(4)) #get video frame height

            
            vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0] #init videowriter
            resize_height, resize_width = vid_write_image.shape[:2]
            out_video_name = f"{source.split('/')[-1].split('.')[0]}"

            while(cap.isOpened): #loop until cap opened or video not complete
            
                print("Frame {} Processing".format(frame_count+1))

                ret, frame = cap.read()  #get frame and success from video capture
                
                if ret: #if success is true, means frame exist

                    if frame_count%2 == 0 and frame_count < 40:
                        orig_image = frame #store frame
                        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) #convert frame to RGB
                        image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                        image_ = image.copy()
                        image = transforms.ToTensor()(image)
                        image = torch.tensor(np.array([image.numpy()]))
                    
                        image = image.to(device)  #convert image data to device
                        image = image.float() #convert image to float precision (cpu)
                        start_time = time.time() #start time for fps calculation
                    
                        with torch.no_grad():  #get predictions
                            output_data, _ = model(image)


                        output_data = non_max_suppression_kpt(output_data,   #Apply non max suppression
                                                    0.25,   # Conf. Threshold.
                                                    0.65, # IoU Threshold.
                                                    nc=model.yaml['nc'], # Number of classes.
                                                    nkpt=model.yaml['nkpt'], # Number of keypoints.
                                                    kpt_label=True)


                        output = output_to_keypoint(output_data)
                        print(output)


                        im0 = image[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
                        im0 = im0.cpu().numpy().astype(np.uint8)
                        
                        im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) #reshape image format to (BGR)
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                        for i, pose in enumerate(output_data):  # detections per image
                            if len(output_data):  #check if no pose
                                for c in pose[:, 5].unique(): # Print results
                                    n = (pose[:, 5] == c).sum()  # detections per class
                                    print("No of Objects in Current Frame : {}".format(n))
                                
                                for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])): #loop over poses for drawing on frame
                                    c = int(cls)  # integer class
                                    kpts = pose[det_index, 6:]
                                    kpts_without_head_kp = pose[det_index, 21:]
                                    label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')

                                    # plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True), 
                                    #             line_thickness=opt.line_thickness,kpt_label=True, kpts=kpts, steps=3, 
                                    #             orig_shape=im0.shape[:2])

                                    # tworzy polaczenia dla nog i rak tylko
                                    plot_skeleton_without_head(im0, kpts_without_head_kp, steps=3, orig_shape=im0.shape[:2])
                                    # wyciagamy wspolrzedne z klatki
                                    frame_cords.append(frame_values(frame_count, kpts_without_head_kp, 3))
                                    
                        results_img.append(im0)    
                        
                        end_time = time.time()  #Calculatio for FPS
                        fps = 1 / (end_time - start_time)
                        total_fps += fps
                        frame_count += 1
                        
                        fps_list.append(total_fps) #append FPS in list
                        time_list.append(end_time - start_time) #append time in list
                        
                        # Stream results
                    else:
                        frame_count += 1

                else:
                    break

            cap.release()
            # cv2.destroyAllWindows()
            avg_fps = total_fps / frame_count
            print(f"Average FPS: {avg_fps:.3f}")
            
    
            p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-r', '24', '-i', '-', '-vcodec', 'mpeg4', '-qscale', '5', '-r', '10', f'{output_img_dir}/{file_name}.avi'], stdin=PIPE)
            for img in results_img:
                im = Image.fromarray(img[:,:,::-1])
                im.save(p.stdin, 'JPEG')
            p.stdin.close()
            p.wait()

            #plot the comparision graph
            plot_fps_time_comparision(time_list=time_list,fps_list=fps_list)

            # tworze csv
            print(frame_cords)
            create_csv(frame_cords, output_csv_dir, file_name)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='football1.mp4', help='video/0 for webcam') #video source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   #device arugments
    parser.add_argument('--view-img', action='store_true', help='display results')  #display results
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') #save confidence in txt writing
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)') #box linethickness
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels') #box hidelabel
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences') #boxhideconf
    opt = parser.parse_args()
    return opt

#function for plot fps and time comparision graph
def plot_fps_time_comparision(time_list,fps_list):
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS and Time Comparision Graph')
    plt.plot(time_list, fps_list,'b',label="FPS & Time")
    plt.savefig("FPS_and_Time_Comparision_pose_estimate.png")

def create_csv(values, output_dir, file_name):
    header = ['Frame nr', "LEFT_SHOULDER", "RIGH_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIPS", "RIGHT_HIPS", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", 
          "LEFT_ELBOW_ANGLE", "RIGHT_ELBOW_ANGLE", "LEFT_KNEE_ANGLE", "RIGHT_KNEE_ANGLE"]
    csv_dir = output_dir / f'{file_name}.csv'
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
