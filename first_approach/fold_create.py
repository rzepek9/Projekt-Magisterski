import pandas as pd
from pathlib import Path
import numpy as np

database_path = Path('/home/s175668/raid/Praca-Magisterska/dataset/pose3d/')
json_files = database_path.glob('*under/*.json')

files = []
for file in json_files:
    print(file)
    file_name = str(file).rsplit('/', 1)[-1]
    camera = str(file_name).rsplit('_', 2)[-2]
    label = int(file_name[-6])
    shoter = str(file_name).split('_', 2)[1]
    if int(file_name.split('_')[0]) in [574, 682, 1307, 1341, 3661, 3248, 4419, 3194, 2829, 4416, 3265, 4213, 4347, 2931,
                                        3678]:
        continue
    files.append([file_name, shoter, camera, label])
    
df = pd.DataFrame(files, columns=['name', 'shoter', 'camera', 'label'])

train = []
val = []
shooter = ['adam', 'krystian', 'kuba', 'gustaw']
for s in shooter:
    for i in range(2):
        shooter_df = df.loc[(df['shoter'] == s) & (df['label'] == i) & (df['camera']=='under')]
        train.extend(shooter_df.iloc[int(len(shooter_df)*0.15):].values)
        val.extend(shooter_df.iloc[:int(len(shooter_df)*0.15)].values)


import random
random.shuffle(train)
random.shuffle(val)

train_df = pd.DataFrame(train, columns = ['name', 'shoter', 'camera', 'label'])
val_df = pd.DataFrame(val, columns = ['name', 'shoter', 'camera', 'label'])

train_df.to_csv('/home/s175668/raid/Praca-Magisterska/dataset/pose3d/train_under_ds.csv', index=False)
val_df.to_csv('/home/s175668/raid/Praca-Magisterska/dataset/pose3d/val_under_ds.csv', index=False)