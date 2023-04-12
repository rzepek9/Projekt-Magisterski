from pathlib import Path
import os
import csv

under_dir = Path('/home/s175668/raid/Praca-Magisterska/four_seconds/site')

file_list = os.listdir(under_dir)

header = ['kuba', '%', 'gustaw', '%', 'tomek',  '%', 'suma']
values = [[0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0]]

for file in file_list:
    if "DS_Store" in str(file):
        continue

    if 'kuba' in str(file):
        n = 0
    elif 'gustaw' in str(file):
        n = 2
    elif 'tomek' in str(file):
        n = 4

    target = str(file)[-5]
    if target == '0':
        i = 0
    else:
        i = 1

    values[i][n] += 1
    values[i][n+1] += 1
    values[i][6] += 1

values[0][1] = values[0][1] / values[0][6] * 100
values[0][3] = values[0][3] / values[0][6] * 100
values[0][5] = values[0][5] / values[0][6] * 100
values[1][1] = values[1][1] / values[1][6] * 100
values[1][3] = values[1][3] / values[1][6] * 100
values[1][5] = values[1][5] / values[1][6] * 100

print(values)
with open('/home/s175668/raid/Praca-Magisterska/Repozytorium/Projekt-Magisterski/data_balance.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(values)
