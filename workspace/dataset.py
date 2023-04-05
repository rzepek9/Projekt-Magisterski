import os
import pandas
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np

# /Users/jakubrzepkowski/Documents/Projekt-magisterski/four_seconds/site/482_tomek_side_0.mov

class FeatureDataset(Dataset):

    def __init__(self, fold_dir):
        self.csv_under_dir = Path(fold_dir) / 'under'
        self.csv_site_dir = Path(fold_dir) / 'site'

        self.file_list = os.listdir(self.csv_site_dir)
        self.x = []
        self.y = []
        
        for file in self.file_list:
            print(self.csv_site_dir/ file)
            # df_under = pd.read_csv(self.csv_under_dir/ file)
            df_site = pd.read_csv(self.csv_site_dir/ file)
            # x_under = df_under.iloc[:, :-1].values
            x_site = df_site.iloc[:,:-5].values

            # x_under_hips_x = df_under.iloc[0,12]
            # x_under_hips_y = df_under.iloc[0,13]

            x_site_hips_x = df_site.iloc[0,12]
            x_site_hips_y = df_site.iloc[0,13]

            # x_under_substract = np.array([x_under_hips_x if i%2==0 else x_under_hips_y for i in range(x_under.shape[1])] for i in range(x_under.shape[0]))
            x_site_substract = np.array([[x_site_hips_x if i%2==0 else x_site_hips_y for i in range(x_site.shape[1])] for i in range(x_site.shape[0])])

            # under = np.subtract(x_under, x_under_substract)
            site = np.subtract(x_site_substract, np.array(x_site))
            print(site)



            self.x.append(site)
            self.y.append(float(df_site.iloc[0,-1]))
    
        self.x_train =torch.tensor(np.array(self.x), dtype=torch.float32)
        self.y_train =torch.tensor(np.array(self.y), dtype=torch.float32)
        print(self.x_train.shape)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx] 