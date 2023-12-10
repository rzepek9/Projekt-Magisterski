import os
import torch
import pandas as pd
import numpy as np
import math
import random

from torch.utils.data import Dataset
from pathlib import Path


class FeatureDataset(Dataset):

    def __init__(self, csv_path):
        
        self.df = pd.read_csv(csv_path)
        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_csv = self.df.iloc[idx].item()
        
        if sample_csv == '/home/s175668/raid/Praca-Magisterska/csv_extracted/conf_th_60/411_tomek_under_0.csv':
            sample_csv = '/home/s175668/raid/Praca-Magisterska/csv_extracted/conf_th_60/411_tomek_under_1.csv'
        
        if sample_csv == '/home/s175668/raid/Praca-Magisterska/csv_extracted/conf_th_60/766_krystian_under_1.csv':
            sample_csv = '/home/s175668/raid/Praca-Magisterska/csv_extracted/conf_th_60/766_krystian_under_0.csv'
            
        try:
            data = pd.read_csv(sample_csv)
        except:
            print(f"{sample_csv}")


        x = data.iloc[:, :-1].values

        started_frame = 0
        check = False
        for i, frame in enumerate(data.values):
            if frame[0] == 0 and frame[1] == 0:
                x[i] = 0
            elif not check:
                hips_x = frame[12]
                hips_y = frame[13]
                y = int(frame[-1])
                started_frame = i
                check = True

        norm_array = np.array([hips_x if i % 2 == 0 else hips_y for i in range(x.shape[1])])

        for i in range(started_frame, 14):
            number_of_zeros = len(np.where(x[i, :] == 0)[0])
            if number_of_zeros > 1:
                continue
            else:
                x[i] = x[i] - norm_array

        x = torch.tensor(np.array(x), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32)

        return x, y
        


class TwoKeypointsChannel(Dataset):
    def __init__(self, fold_dir):
        self.csv_under_dir = Path(fold_dir) / 'under'
        self.csv_site_dir = Path(fold_dir) / 'site'

        self.files = os.listdir(self.csv_under_dir)


    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        csv_file = self.csv_under_dir / self.files[idx]

        df = pd.read_csv(csv_file)
        y = float(df.iloc[0, -1])

        x_hips_x = df.iloc[0, 12]
        x_hips_y = df.iloc[0, 13]

        x_vectors = df.iloc[:, :-1].values

        substract = np.array([[x_hips_x if i % 2 == 0 else x_hips_y for i in range(x_vectors.shape[1])] 
                              for _ in range(x_vectors.shape[0])])
        
        x_vectors = np.subtract(substract, x_vectors)



        for frame, val in enumerate(x_vectors):
            if frame == 0:
                array = np.array([[val[i*2], val[i*2+1]] for i in range(12)])
                print(array.shape)
            else:
                array_to_stack = np.array([[val[i*2], val[i*2+1]] for i in range(12)])
                array = np.dstack((array, array_to_stack))

        return array, y
    


class VectorLength(Dataset):

    def __init__(self, csv_path, augment=False, return_kp = False):
        
        self.df = pd.read_csv(csv_path)
        self.df = self.df.reset_index(drop=True)
        self.augment = augment
        self.return_kp = return_kp

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_csv = self.df.iloc[idx].item()

        try:
            data = pd.read_csv(sample_csv)
        except:
            print(f"{sample_csv}")
    

        y = data.iloc[0,-1]
        conf = data.iloc[:, 4:-1:3].values
        data = data.drop(data.columns[[i for i in range(4, 39, 3)]], axis = 1)
        data = data.drop('TARGET', axis=1)
        cords = data.iloc[:, :].values

        conf = interpolation(conf)
        cords = interpolation(cords)

        nose_x, nose_y = cords[0, 0], cords[0, 1]
        cords_x, cords_y = cords[:, 2::2], cords[:, 3::2]

        lh_x, rh_x = cords_x[0,6], cords_x[0,7]
        lh_y, rh_y = cords_y[0,6], cords_y[0,7]
        x0, y0 = abs(lh_x - rh_x)/2, abs(lh_y - rh_y)/2

        x0 += min(lh_x, rh_x)
        y0 += min(lh_y, rh_y)

        normalize_vector = math.sqrt((nose_x-x0)**2+(nose_y-y0)**2)

        vector_length = calulate_vector_length(cords_x, cords_y, x0, y0, cords) / normalize_vector

        if self.augment:
            prob = random.random()
            if prob > 0.5:
                noise = np.random.normal(loc=0, scale=0.1, size=vector_length.shape)
                vector_length = vector_length + noise

        if self.return_kp: 
            cords = cords / normalize_vector
            x = torch.tensor(cords[:,2:], dtype=torch.float32)
        else:
            x = torch.tensor(vector_length, dtype=torch.float32)

        y = torch.tensor(y, dtype=torch.float32)


        return x, y
    

def interpolation(cords):
    index = np.where(cords == 0)
    # Cords shape: (timestep, cords_values)
    for c in range(cords.shape[1]):
        # Take signal of one joint
        cords_to_interp = cords[:, c]

        timestep = np.where(cords_to_interp > 0)[0]
        index_to_interp = np.where(cords_to_interp == 0)[0]
        cords_to_interp = np.delete(cords_to_interp, index_to_interp)
        if len(index_to_interp) ==0 :
            continue
        
        interpolated = np.interp(index_to_interp, timestep, cords_to_interp)
        
        for i, index in enumerate(index_to_interp):
            cords[index, c] = interpolated[i]
        
    return cords

def calulate_vector_length(cords_x, cords_y, x0, y0, cords):
    vector_lenth = np.zeros((14,12))
    for t in range(14):
        # nose_x, nose_y = cords[t, 0], cords[t, 1]
        # lh_x, rh_x = cords_x[t,6], cords_x[t,7]
        # lh_y, rh_y = cords_y[t,6], cords_y[t,7]

        # x0, y0 = abs(lh_x - rh_x)/2, abs(lh_y - rh_y)/2

        # x0 += min(lh_x, rh_x)
        # y0 += min(lh_y, rh_y)

        # normalize_vector = math.sqrt((nose_x-x0)**2+(nose_y-y0)**2)

        for kp in range(12):
            x1, y1 = cords_x[t, kp], cords_y[t,kp]
            vector_lenth[t, kp] = math.sqrt((x1-x0)**2+(y1-y0)**2) #/ normalize_vector
    
    return vector_lenth
