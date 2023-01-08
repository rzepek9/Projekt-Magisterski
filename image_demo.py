import cv2
import time
import argparse
import os
import torch
import numpy as np
from PIL import Image
from subprocess import Popen, PIPE

import posenet


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()


def main():
    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    



    filenames = [
        f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

    start = time.time()
    for video in os.listdir('clip/'):
        print(video)
        if video == '.DS_Store':
            continue
        results_img = []
        frame_list = []
        cap = cv2.VideoCapture(f'clip/{video}')
        success, image = cap.read()
        while success:
            frame_list.append(image)
            success, image = cap.read()

        for f in frame_list[:-1:3]:
            input_image, draw_image, output_scale = posenet.read_imgfile(
                f, scale_factor=args.scale_factor, output_stride=output_stride)

            with torch.no_grad():
                input_image = torch.Tensor(input_image).cuda()

                heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

                pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                    heatmaps_result.squeeze(0),
                    offsets_result.squeeze(0),
                    displacement_fwd_result.squeeze(0),
                    displacement_bwd_result.squeeze(0),
                    output_stride=output_stride,
                    max_pose_detections=1,
                    min_pose_score=0.25)

            keypoint_coords *= output_scale

            draw_image = posenet.draw_skel_and_kp(
                draw_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.25, min_part_score=0.25)

        
            string = []
            if not args.notxt:
                
                for pi in range(len(pose_scores)):
                    if pose_scores[pi] == 0.:
                        break
                    for ki, (s, c) in enumerate(zip(keypoint_scores[0, 5:], keypoint_coords[0, 5:, :])):
                        string.append(f'Keypoint {posenet.PART_NAMES[ki+5]}, score = {round(s, 2)}, coord = {str(np.around(c,2))}')    

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.4
            color = (255, 0, 0)
            thickness = 1

            for i, line in enumerate(string):
                draw_image = cv2.putText(draw_image, line, (10, 12 + 12*i), font, 
                            fontScale, color, thickness, cv2.LINE_AA)

            results_img.append(draw_image)


        fps, duration = 24, 100
        p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-r', '24', '-i', '-', '-vcodec', 'mpeg4', '-qscale', '5', '-r', '10', f'output/{video[:-4]}.avi'], stdin=PIPE)
        for img in results_img:
            im = Image.fromarray(img[:,:,::-1])
            im.save(p.stdin, 'JPEG')
        p.stdin.close()
        p.wait()
        # fourcc = cv2.VideoWriter_fourcc(*'MJPEG')
        # out = cv2.VideoWriter(f'output/{video[:-4]}_kp.avi',cv2.VideoWriter_fourcc(*'MJPG'), 10, (480,640))

        # print(len(results_img))
        # for  img in results_img:
        #     out.write(img)
        
        # out.release()
        print('Average FPS:', len(filenames) / (time.time() - start))


if __name__ == "__main__":
    main()
