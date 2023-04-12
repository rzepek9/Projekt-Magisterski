import os
import torch
import pandas as pd
import numpy as np


from torch.utils.data import Dataset
from pathlib import Path


class FeatureDataset(Dataset):

    def __init__(self, fold_dir):
        self.csv_under_dir = Path(fold_dir) / 'under'
        self.csv_site_dir = Path(fold_dir) / 'site'

        self.file_list = os.listdir(self.csv_site_dir)
        self.x = []
        self.y = []

        for file in self.file_list:

            # Wczytanie danych z csv
            df_under = pd.read_csv(self.csv_under_dir / file)
            x_under = df_under.iloc[:, :-1].values

            df_site = pd.read_csv(self.csv_site_dir / file)
            x_site = df_site.iloc[:, :-5].values

            # Wartosci kp lewego biodra
            x_under_hips_x = df_under.iloc[0, 12]
            x_under_hips_y = df_under.iloc[0, 13]

            x_site_hips_x = df_site.iloc[0, 12]
            x_site_hips_y = df_site.iloc[0, 13]

            # # Arraye do normalizacji do jednego ukladu wspolrzedncych gdzie miejscem (0,0) jest lewe biodro z pierwszej klatki
            x_under_substract = np.array([[x_under_hips_x if i % 2 == 0 else x_under_hips_y for i in range(
                x_under.shape[1])] for _ in range(x_under.shape[0])])
            x_site_substract = np.array([[x_site_hips_x if i % 2 == 0 else x_site_hips_y for i in range(x_site.shape[1])]
                                         for _ in range(x_site.shape[0])])

            # Normalizacja
            under = np.subtract(x_under, x_under_substract)
            site = np.subtract(x_site_substract, np.array(x_site))

            self.x.append(site)
            self.y.append(float(df_site.iloc[0, -1]))
            self.x.append(under)
            self.y.append(float(df_under.iloc[0, -1]))

        self.x_train = torch.tensor(np.array(self.x), dtype=torch.float32)
        self.y_train = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
