import subprocess
from pathlib import Path
import os

path = Path('/home/s175668/raid/Praca-Magisterska/dataset/preprocessed/side')
json_p = os.listdir('/home/s175668/raid/Praca-Magisterska/dataset/pose2d/side')
files = path.glob('*')

for file in files:
    # file_name = str(file).rsplit('/', -1)[-1][:-3] + 'json'
    # if file_name in json_p:
    #     continue

    print(file)
    p = subprocess.call(f"python /home/s175668/raid/Praca-Magisterska/Repozytorium/mmpose/demo/inferencer_demo.py {file} --pose2d human --pred-out-dir /home/s175668/raid/Praca-Magisterska/dataset/pose2d/check --device cuda:3", shell=True)