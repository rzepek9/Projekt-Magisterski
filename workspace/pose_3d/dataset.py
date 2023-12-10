import os
import torch
import pandas as pd
import numpy as np
import math
import random
import json

from torch.utils.data import Dataset
from pathlib import Path
import sklearn.preprocessing as sk


class KeypointsMatrix(Dataset):

    def __init__(self, csv_path, camera=None, cordinates_move=False, choose_points=True, height_normalize=False,
                 signal_normalize_0_1=False, cg=False, vf=False, shoter=None, aug=False):
        data = pd.read_csv(csv_path)
        self.data = data.reset_index(drop=True)
        
        self.path_3d = Path('/home/s175668/raid/Praca-Magisterska/dataset/pose3d/')
        self.path_2d = Path('/home/s175668/raid/Praca-Magisterska/dataset/pose2d/')
        
        self.cordinates_move = cordinates_move
        self.choose_kpt = choose_points
        self.height_normalize = height_normalize
        self.signal_normalize_0_1 = signal_normalize_0_1
        
        self.cords_features = cg or vf
        self.CG = cg
        self.velocity_features = vf
        self.augment = aug
        
        self.normalize_dict = {'adam': 181,
                               'krystian': 192,
                               'kuba': 175,
                               'gustaw': 182}
        
        if camera is not None:
            self.data = self.data[self.data['camera'] == camera]
        
        if shoter is not None:
            self.data = self.data[self.data['shoter'] == shoter]
        

    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, idx):
        file_name, shoter, camera, label= self.data.iloc[idx].values
        interpolated = False
        file_name = f'{file_name}.json'
        
        with open(self.path_3d / camera / file_name) as f:
                json_3d = json.load(f)
       
        with open(self.path_2d / camera / file_name) as f:
            json_2d = json.load(f)
       
       
        all_frames = []
        for frame in json_2d:
            frame_id = frame['frame_id']
            pose_2d = frame['instances'][0]
            
            if pose_2d['bbox_score'] < 0.5  or pose_2d['bbox_score'] == 1.0:
                all_frames.append(np.full((17,3), np.nan))
                interpolated = True
            else:
                assert len(json_3d[frame_id]['instances']), f"{file_name} too many instances"
                frame_cords = np.array(json_3d[frame_id]['instances'][0]['keypoints'])
                all_frames.append(frame_cords)
            
        cords = np.stack(all_frames, 2)
        
        if self.cordinates_move:
            cords = joint_vector_cordinates(cords)
            
        #all_frames shape(17, 3, 14)
        if interpolated:
            cords = interpolation(cords)
            
        if self.augment > 0:
            if np.random.uniform(0., 1.) < 0.3:
                for kp in range(cords.shape[0]):
                    for ch in range(cords.shape[1]):
                        cords[kp, ch, :] += np.random.normal(cords[kp,ch,:].mean(), cords[kp, ch, :].std(), cords[kp, ch, :].shape) * self.augment
        
        
        if self.cords_features:
            cords = cords.transpose(2, 1, 0)
            if self.CG and self.velocity_features:
                cords_feature = CG_normalize(cords)
                velocity_feature = velocity_features(cords)
                
                features = np.concatenate((cords_feature, velocity_feature), axis=1)

            elif self.CG:
                features = CG_normalize(cords)
            
            elif self.velocity_features:
                features = velocity_features(cords)
                
            if self.signal_normalize_0_1:
                features = signal_normalize(features)
            
            return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
                
        else:    
            #all_frames shape(kp, ch, frames) -> shape(frames, ch, kp)
            cords = choose_keypoints(cords)
            cords = cords.transpose(2, 1, 0)
            
            if self.height_normalize:
                cords = cords / self.normalize_dict[shoter]
                
            cords = cords.reshape(18, cords.shape[1]*cords.shape[2])
            
            if self.signal_normalize_0_1:
                cords = signal_normalize(cords)
                
            assert cords.shape[0] == 18, file_name
        
            return torch.tensor(cords, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        
    

def interpolation(cords):
    # Inputs cords shape (17, 3, 14) - cord_ch, frames, keypoint
    for keypoint in cords:
        for channel in keypoint:
            nans, x = np.isnan(channel), lambda z: z.nonzero()[0]
            channel[nans]= np.interp(x(nans), x(~nans), channel[~nans])
    return cords

def joint_vector_cordinates(cords):
    cords = cords.transpose(2, 1, 0)
    for frame in cords:
        for channel in frame:
            p_0_0 = channel[0]
            channel -= p_0_0
    cords = cords.transpose(2, 1, 0)
    return cords


def choose_keypoints(cords, ax = 0):
    indicies = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]
    cords = np.take(cords, indicies, axis = ax)
    return cords

def signal_normalize(cords):
    cords = cords.transpose(1, 0)
    for i, signal in enumerate(cords):
        cords[i] = (signal - signal.min()) / (signal.max() - signal.min())
    
    cords = cords.transpose(1, 0)
    return cords

# https://openaccess.thecvf.com/content_WACVW_2020/papers/w5/N._Exploring_Techniques_to_Improve_Activity_Recognition_using_Human_Pose_Skeletons_WACVW_2020_paper.pdf
def CG_normalize(cords):
    new_cords = cords.copy()
    cg_features = []
    for i, frame in enumerate(cords):
        cg = [frame[i].mean() for i in range(3)]
        d = max_vector_length(frame)
        
        for ch in range(3):
            new_cords[i, ch] = (new_cords[i, ch] - cg[ch]) / d
            
        selected_kp = choose_keypoints(new_cords[i], ax=1)
        cg_features.append(selected_kp.flatten())
        
    cg_features = np.stack(cg_features, axis=0)
    return cg_features

def velocity_features(cords):
    velocity_features = []
    for i, frame in enumerate(cords):
        d = max_vector_length(frame)
        
        if i == 0:
            vf_features = (frame / d)
        else:
            vf_features = ((frame - cords[i-1]) / d)
        
        selected_kp = choose_keypoints(vf_features, ax=1)
        velocity_features.append(selected_kp.flatten())
    
    velocity_features = np.stack(velocity_features, axis=0)
    return velocity_features


def max_vector_length(frame):
    reference_joint = frame[:, 3]
    check_joints = [16, 14, 10]
    
    longest_distance = np.array([calculate_vector(reference_joint, frame[:, j]) for j in check_joints]).max()
    
    return longest_distance

def calculate_vector(joint_0, joint_1):
    vector_length = math.sqrt((joint_0[0] - joint_1[0])**2 +
                            (joint_0[1] - joint_1[1])**2 +
                            (joint_0[2] - joint_1[2])**2)
    return vector_length

if __name__ == '__main__':
    kp = KeypointsMatrix('/home/s175668/raid/Praca-Magisterska/dataset/folds/made/train_made_balance.csv', cg=True, vf=True)
    kp[-1]
