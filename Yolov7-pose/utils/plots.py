# Plotting utils
import math
from numpy import sqrt
import cv2


KEYPOINTS = ["NOSE", "LEFT_EYE", "RIGH_EYE", "LEFT_EAR", "RIGH_EYE", "LEFT_SHOULDER", "RIGH_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
              "LEFT_HIPS", "RIGHT_HIPS", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"]
KEYPOINTS_INTERESTED = ["LEFT_SHOULDER", "RIGH_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIPS", "RIGHT_HIPS", "LEFT_KNEE", "RIGHT_KNEE", 
                        "LEFT_ANKLE", "RIGHT_ANKLE"]


def plot_skeleton_without_head(im, kpts, steps):
    connection = [[0, 2], [1, 3], [2, 4], [3, 5], [6, 8], [7, 9], [8, 10], [9,11]]

    for i in range(len(KEYPOINTS_INTERESTED)):
        x_cord, y_cord, conf_cord = kpts[i*steps], kpts[i*steps+1], kpts[i*steps+2]

        if conf_cord > 0.4:
            cv2.circle(im, (int(x_cord), int(y_cord)), 3, (0,0,0), 1)
    
    for c in connection:
        pos1 = (int(kpts[c[0]*steps]) , int(kpts[c[0]*steps+1]))
        pos2 = (int(kpts[c[1]*steps]) , int(kpts[c[1]*steps+1]))

        cv2.line(im, pos1, pos2, (255,255,255), thickness=1)

        
def calculate_angles(kpts,steps):
    # MOCNO DO POPRAWY TEN CALCULATE ANGLE
    angles = [[0,2,4], [1,3,5], [6,8,10], [7,9,11]]

    for i, angle in enumerate(angles):
        pos1 = (float(kpts[angle[0]*steps]) , float(kpts[angle[0]*steps+1]))
        pos2 = (float(kpts[angle[1]*steps]) , float(kpts[angle[1]*steps+1]))
        pos3 = (float(kpts[angle[2]*steps]) , float(kpts[angle[2]*steps+1]))

        vec_1 = sqrt((pos1[0]- pos2[0])**2 + (pos1[1]-pos2[1])**2)
        vec_2 = sqrt((pos3[0]- pos2[0])**2 + (pos3[1]-pos2[1])**2)

        calculated_angle = vec_1/vec_2

        if i == 0:
            left_elbow = math.atan(calculated_angle)
        elif i == 1:
            right_elbow = math.atan(calculated_angle)
        elif i == 2:
            left_knee = math.atan(calculated_angle)
        elif i == 3:
            right_knee = math.atan(calculated_angle)
    
    dg = [math.degrees(left_elbow), math.degrees(right_elbow), math.degrees(left_knee), math.degrees(right_knee)]
    for i, deg in enumerate(dg):
        if deg < 0:
            dg[i] += 360

    dg = [0 if math.isnan(x) else x for x in dg]
    return dg


def frame_values(kpts, steps, target, calculate=None):
    value = []

    for cord in range(len(KEYPOINTS_INTERESTED)):
        value.append(float(kpts[cord*steps])/640)
        value.append(float(kpts[cord*steps+1])/480)
    
    if calculate:
        angles = calculate_angles(kpts, steps)
        for a in angles:
            value.append(a)
    
    value.append(target)
    return value

    





