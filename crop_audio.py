from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pathlib import Path

# endtime poszczegolne czasy do wyciecia przykladowo start 0s koniec 7s, nastepne okno do wyciecia bedzie od 7s do nastepnego czasu
# last clip number oznacza ostatni wyciety w poprzednim wideo clip
#  nazwa clipu video['destination_path']+f'{video["shooter"]}_{video["camera"]}_{str(video["last_clip_number"])}_{video["target"][i]}.mov'

input_dir = Path('/Users/jakubrzepkowski/Documents/Projekt-magisterski/Wideo/KOSZ')
output_dir = Path('/Users/jakubrzepkowski/Documents/Projekt-magisterski/Clipy_wideo')


video = {
    'path_under': str(input_dir /'kosz1130.avi'),
    'path_site': str(input_dir /'test11_30_ja.avi'),
    'end_time': [(4, 'celny'), (8, 'out'), (20, 'niecelny'), (24, 'celny'), (29, 'niecelny'), (34, 'niecelny'), (40, 'niecelny'), (44, 'out'), (53, 'niecelny'), (59, 'celny'), (64, 'niecelny'), (68, 'niecelny'), (74, 'niecelny'), (80, 'celny'), (84, 'celny'), (90, 'celny'), (95, 'celny'), (102, 'celny'), (108, 'niecelny'), (113, 'out')],
    'target': ['celny', 'out', 'niecelny', 'celny',  'niecelny',  'celny',  'niecelny',  'niecelny',  'niecelny',  'celny',  'niedolot',  'celny',  'niedolot', 'celny', 'celny', 'celny', 'celny', 'niecelny', 'celny', 'niecelny', 'niecelny', 'niecelny', 'out', 'celny', 'celny',  'niecelny', 'niecelny', 'celny', 'niecelny', 'celny', 'niecelny', 'celny', 'niecelny', 'celny', 'out','celny', 'celny','out', 'niecelny', 'niecelny', 'niecelny', 'celny', 'celny', 'niecelny', 'niecelny', 'niecelny', 'celny', 'celny', 'niecelny', 'celny', 'celny', 'niecelny','celny','celny','celny', 'celny',  'celny', 'out'],
    'camera_under': 'under',
    'camera_site': 'site',
    'shooter': 'kuba',
    'last_clip_number': 111,
    'destination_path': output_dir
}
number_of_clips = len(video['end_time']) - 1
# assert len(video['end_time']) == len(video['target']), 'To much target or times'
print((video['end_time'][1][0]))
print((video['target'][45:]))

for i, target in enumerate(video['end_time']):
    if (target[1] != 'out'):
        video['last_clip_number'] += 1
        ffmpeg_extract_subclip(video['path_site'], video['end_time'][i][0], video['end_time'][i+1][0], 
            targetname= output_dir /'site'/ f'{str(video["last_clip_number"])}_{video["shooter"]}_{video["camera_site"]}_{target[1]}.mov')
        ffmpeg_extract_subclip(video['path_under'],video['end_time'][i][0], video['end_time'][i+1][0], 
            targetname= output_dir /'under'/ f'{str(video["last_clip_number"])}_{video["shooter"]}_{video["camera_under"]}_{target[1]}.mov')
        



