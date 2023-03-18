from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pathlib import Path

# endtime poszczegolne czasy do wyciecia przykladowo start 0s koniec 7s, nastepne okno do wyciecia bedzie od 7s do nastepnego czasu
# last clip number oznacza ostatni wyciety w poprzednim wideo clip
#  nazwa clipu video['destination_path']+f'{video["shooter"]}_{video["camera"]}_{str(video["last_clip_number"])}_{video["target"][i]}.mov'

input_dir = Path('/Users/jakubrzepkowski/Documents/Projekt-magisterski/Wideo/kosz')
output_dir = Path('/Users/jakubrzepkowski/Documents/Projekt-magisterski/Clipy_wideo')


video = {
    'path_under': str(input_dir /'kosz122.avi'),
    'path_site': str(input_dir /'test12_2.avi'),
    'end_time': [(2.6, 'celny'), (5, 'out'), (8, 'niecelny'), (12, 'celny'), (17, 'niecelny'), (22, 'celny'), (28, 'niecelny'), (33.3, 'celny'), (38, 'celny'), (43, 'celny'), (48, 'celny'), (53, 'out'), (55, 'niecelny'), (60, 'niecelny'), (64, 'out'), (65, 'celny'), (69, 'out'), (1*60+12, 'celny'), (76, 'out'), (77, 'celny'), (81, 'niecelny'), (85, 'out'), (90, 'celny'), (95, 'out'), (98, 'niecelny'), (103, 'niecelny'), (107, 'niecelny'), (113, 'out'), (2*60+55, 'celny'), (3*60, 'celny'), (3*60+5, 'niecelny'), (3*60+10, 'niecelny'), (3*60+15, 'niecelny'), (3*60+19, 'niecelny'), (3*60+25, 'celny'), (3*60+31, 'niecelny'), (3*60+36, 'celny'), (3*60+42, 'niecelny'), (3*60+47, 'celny'),(3*60+51, 'out'), (3*60+53, 'celny'),(3*60+58, 'niecelny'), (4*60+3, 'niecelny'), (4*60+8, 'celny'), (4*60+13, 'celny'), (4*60+18, 'out'), (4*60+21, 'niecelny'), (4*60+26, 'out'),(4*60+27, 'celny'), (4*60+32, 'out'), (5*60+17, 'niecelny'), (5*60+21, 'out'), (5*60+22, 'celny'), (5*60+26, 'out'),(5*60+29, 'celny'), (5*60+34, 'out'), (5*60+35, 'niecelny'), (5*60+40, 'celny'), (5*60+45, 'celny'), (5*60+50, 'celny'), (5*60+55, 'celny'), (6*60, 'celny'), (6*60+5, 'out'), (6*60+7, 'niecelny'), (6*60+12, 'out'), (6*60+14, 'celny'), (6*60+18, 'out'), (6*60+21, 'celny'), (6*60+25, 'out'), (6*60+28, 'celny'), (6*60+33, 'niecelny'), (6*60+38, 'out'), (6*60+41, 'celny'), (6*60+46, 'celny'), (6*60+51, 'niecelny'), (6*60+56, 'out'), (6*60+57, 'niecelny'), (7*60+2, 'celny'), (6*60+7, 'out')],
    'target': ['celny', 'out', 'niecelny', 'celny',  'niecelny',  'celny',  'niecelny',  'niecelny',  'niecelny',  'celny',  'niedolot',  'celny',  'niedolot', 'celny', 'celny', 'celny', 'celny', 'niecelny', 'celny', 'niecelny', 'niecelny', 'niecelny', 'out', 'celny', 'celny',  'niecelny', 'niecelny', 'celny', 'niecelny', 'celny', 'niecelny', 'celny', 'niecelny', 'celny', 'out','celny', 'celny','out', 'niecelny', 'niecelny', 'niecelny', 'celny', 'celny', 'niecelny', 'niecelny', 'niecelny', 'celny', 'celny', 'niecelny', 'celny', 'celny', 'niecelny','celny','celny','celny', 'celny',  'celny', 'out'],
    'camera_under': 'under',
    'camera_site': 'site',
    'shooter': 'kuba',
    'last_clip_number': 54,
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
        



