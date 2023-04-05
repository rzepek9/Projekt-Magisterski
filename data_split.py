from pathlib import Path

under_dir = Path('/Users/jakubrzepkowski/Documents/Projekt-magisterski/four_seconds/site')

file_list = under_dir.glob('*')
list_for_split = []

for file in file_list:
    target = str(file)[-5]
    id = str(file).rsplit('/', 1)[-1].split('_')[0]

    if 'kuba' in str(file) and target == '1':
        list_for_split.append(str(file))

with open('/Users/jakubrzepkowski/Documents/Projekt-magisterski/Repozytorium/Projekt-Magisterski/test/fold0.txt', 'a') as f:
    f.writelines('\n'.join(list_for_split[:29]))

with open('/Users/jakubrzepkowski/Documents/Projekt-magisterski/Repozytorium/Projekt-Magisterski/train/fold0.txt', 'a') as f:
    f.writelines('\n'.join(list_for_split[29:]))
