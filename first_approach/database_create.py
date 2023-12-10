from pathlib import Path
import csv

files_dir = Path('/home/s175668/raid/Praca-Magisterska/csv_extracted/th_60_kp_with_conf')

files_list = files_dir.glob('*.csv')

csv_info = []
for file in files_list:
    target = int(str(file)[-5])
    file_name = str(file).rsplit('/', 1)[-1]
    shooter = file_name.split('_', 2)[1]
    camera = file_name.split('_', 3)[2]
    values = [file, file_name, shooter, camera, target]
    csv_info.append(values)

header = ['path', 'file_name', 'shooter', 'camera', 'label']
print(csv_info)

with open('/home/s175668/raid/Praca-Magisterska/csv_extracted/th_60_kp_with_conf/database.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(csv_info)


