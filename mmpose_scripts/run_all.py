import subprocess
from pathlib import Path
import os

path = Path('/home/s175668/raid/Praca-Magisterska/dataset/preprocessed/side')
json_p = os.listdir('/home/s175668/raid/Praca-Magisterska/dataset/pose3d/side')
files = path.glob('*')

for file in files:
    # file_name = str(file).rsplit('/', -1)[-1][:-3] + 'json'
    # if file_name in json_p:
    #     continue

    print(file)
    p = subprocess.call(f"python /home/s175668/raid/Praca-Magisterska/Repozytorium/mmpose/demo/inferencer_demo.py {file} --pose3d human3d --pred-out-dir /home/s175668/raid/Praca-Magisterska/dataset/pose3d/check --vis-out-dir /home/s175668/raid/Praca-Magisterska/dataset/pose3d_vis/check --device cuda:3", shell=True)