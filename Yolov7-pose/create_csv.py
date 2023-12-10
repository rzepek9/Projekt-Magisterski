import cv2
import torch
import argparse
import csv
import numpy as np

from pathlib import Path
from torchvision import transforms

from utils.datasets import letterbox
from utils.torch_utils import select_device
from utils.general import non_max_suppression_kpt, strip_optimizer
from utils.plots import frame_values
from models.experimental import attempt_load



"""
Działanie pliku 
- flaga source sciezka do fold.txt
- uzyj flagi site oraz train aby wygenerowac csv dla foldu treningowego z bocznej perspektywy, wazne aby umiescic dobra
 sciezke 
- uyztj flagi site oraz test dla foldu testowego z bocznej perspektywy
- analogicnzie dla dolnej perspektywy

Wygenerowany csv posiada kp w stylu xyxyxy... oraz na koncu target w przypadku dolnej perspektywy, bez kp twarzy
Dla bocznej perspektywy kp w tym samym stylu tylko przed targetem dodane zostaly 4 katy (2 w lokciach , 2 w kolanach) 
KATY SA DO POPRAWY
"""



@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt", source='/home/s175668/raid/Praca-Magisterska/four_seconds/under', device='0',
        test=False, train=False, site=False, under=False):

    source_list = Path(source).glob('*.avi')
    device = select_device(opt.device)  # select device
    model = attempt_load(poseweights, map_location=device).eval()  # Load model

    output_csv_dir = Path('/home/s175668/raid/Praca-Magisterska/csv_extracted/th_60_kp_with_conf')
    bad_files = []
    database_info = []
    for clip in source_list:

        file_name = str(clip).rsplit('/', 1)[-1][:-4]
        file_name = file_name.replace('site', 'side')
        side = bool('side' in file_name)

        label = file_name.rsplit('_', 1)[-1]

        if label == 'celny':
            file_name = file_name.replace("celny", "1")
        elif label == 'niecelny':
            file_name = file_name.replace("niecelny", "0")
        
        shooter = file_name.split('_', 2)[1]
        
        frame_features = []
        csv_path = output_csv_dir / f"{file_name}.csv"

        target = int(file_name[-1])

        cap = cv2.VideoCapture(str(clip))
        frame_number = 1

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
                                                      0.60, # Conf. Threshold.
                                                      0.65,  # IoU Threshold.
                                                      nc=model.yaml['nc'],  # Number of classes.
                                                      nkpt=model.yaml['nkpt'], # Number of keypoints.
                                                      kpt_label=True)

                for pose in output_data:
                    # loop over poses for drawing on frame
                    for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:, :6])):
                        # Na pojedynczych filmikach wykrywa osobe pod koszem co wyplywalo na wynik csv
                        if (side and pose[det_index, 21]/640 < 0.35) or created:
                            continue
                        else:
                            nose = pose[det_index, 6:8]
                            kpts = pose[det_index, 21:]

                            frame_features.append(frame_values(kpts, steps=3, target=target, fw=frame_width,
                                                               fh=frame_hight, nose=nose))
                            
                            created = True

                if not created:
                    frame_features.append([0 if i !=38  else target for i in range(39)])
                    bad_files.append([file_name, frame_number])
                frame_number += 1


            else:
                break

        cap.release()
        cv2.destroyAllWindows()

        create_csv(frame_features, csv_path, False)
        database_info.append([csv_path, shooter, target])
        assert len(frame_features) == 14, file_name

    try:
        with open(str(output_csv_dir / 'prob_badfiles.txt'), 'a+', newline='') as f:
            writer = csv.writer(f)
            for element in bad_files:
                print(element)
                writer.writerow(element)
        database_info(output_csv_dir, database_info)
        print(database_info)
    except:
        print(bad_files)
        print("Problem with csv code")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str,
                        default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='', help='fold.txt path')
    parser.add_argument('--device', type=str, default='cpu',
                        help='cpu/0,1,2,3(gpu)')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--site', action='store_true', default=False)
    parser.add_argument('--under', action='store_true', default=False)
    return parser.parse_args()

def databse_info(csv_dir, values):
    header = ['path', 'shooter', 'label']
    with open(str(csv_dir/ "database_info.csv"), 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(values)


def create_csv(values, output_dir, calculate_angle):
    header = ["NOSE_X", "NOSE_Y", "LEFT_SHOULDER_X", "LEFT_SHOULDER_Y", "CONF", "RIGH_SHOULDER_X", "RIGH_SHOULDER_Y",
            "CONF", "LEFT_ELBOW_X", "LEFT_ELBOW_Y","CONF", "RIGHT_ELBOW_X",  "RIGHT_ELBOW_Y", "CONF",
            "LEFT_WRIST_X", "LEFT_WRIST_Y", "CONF", "RIGHT_WRIST_X", "RIGHT_WRIST_Y", "CONF",
            "LEFT_HIPS_X", "LEFT_HIPS_Y", "CONF", "RIGHT_HIPS_X", "RIGHT_HIPS_Y", "CONF",
            "LEFT_KNEE_X", "LEFT_KNEE_Y","CONF", "RIGHT_KNEE_X", "RIGHT_KNEE_Y", "CONF", "LEFT_ANKLE_X", "LEFT_ANKLE_Y",
             "CONF", "RIGHT_ANKLE_X", "RIGHT_ANKLE_Y", "CONF", "TARGET"]

    with open(str(output_dir), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(values)


# main function
def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)
    main(opt)
