import pandas
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np

# /Users/jakubrzepkowski/Documents/Projekt-magisterski/four_seconds/site/482_tomek_side_0.mov

class FeatureDataset(Dataset):

    def __init__(self, fold_txt):
        self.csv_under_dir = Path(fold_txt.rsplit('/',1)[0]) / 'under'
        self.csv_site_dir = Path(fold_txt.rsplit('/',1)[0]) / 'site'

        self.x = []
        self.y = []

        self.Lines = []
        self.file_id = []
        with open(fold_txt, 'r') as file:
            self.Lines = file.readline()
        
        for line in self.Lines:
            file_name = line.rsplit('/', 1)[-1]
            self.file_id.append(file_name.split('_', 1)[0])
        
        for i in self.file_id:
            df_under = pd.read_csv(self.csv_under_dir/f'{i}.csv')
            df_site = pd.read_csv(self.csv_site_dir/f'{i}.csv')
            x_under = df_under[:, :-1].values
            x_site = df_site[:,:-1].valuues

            self.x.append(np.concatenate((x_under, x_site), axis=1))
            self.y.append(df_under[0,-1])
    
        self.x_train =torch.tensor(np.array(self.x), dtype=torch.float32)
        self.y_train =torch.tensor(np.array(self.y))

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx] 