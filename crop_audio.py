from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pathlib import Path

# endtime poszczegolne czasy do wyciecia przykladowo start 0s koniec 7s, nastepne okno do wyciecia bedzie od 7s do nastepnego czasu
# last clip number oznacza ostatni wyciety w poprzednim wideo clip
#  nazwa clipu video['destination_path']+f'{video["shooter"]}_{video["camera"]}_{str(video["last_clip_number"])}_{video["target"][i]}.mov'

input_dir = Path('/Users/jakubrzepkowski/Documents/Projekt-magisterski/Wideo/kosz')
output_dir = Path('/Users/jakubrzepkowski/Documents/Projekt-magisterski/Clipy_wideo')


video = {
    'path_under': str(input_dir /'kosz114.avi'),
    'path_site': str(input_dir /'test11_4.avi'),
    'end_time': [0, 3*60+39, 3*60+47, 3*60+52, 3*60+56, 4*60+1, 4*60+6, 240+11, 240+16 , 240+21, 240+26, 240+31, 240+36, 240+42, 240+47, 240+53, 240+59, 300+4, 300+9, 300+16, 300+20, 300+26, 300+31, 300+36, 420, 420+4, 420+9, 420+15, 420+19, 420+23, 420+28, 420+33, 420+39, 420+44, 420+49,420+53, 420+57, 480+3,  480+7, 480+56, 540+2,540+6,540+11,540+16, 540+21, 540+25, 540+31, 540+36, 540+42, 540+48, 540+53, 540+58, 604, 610, 616, 621, 627, 632,637],
    'target': ['out', 'niecelny', 'celny', 'niecelny', 'celny',  'niecelny',  'celny',  'niecelny',  'niecelny',  'niecelny',  'celny',  'niedolot',  'celny',  'niedolot', 'celny', 'celny', 'celny', 'celny', 'niecelny', 'celny', 'niecelny', 'niecelny', 'niecelny', 'out', 'celny', 'celny',  'niecelny', 'niecelny', 'celny', 'niecelny', 'celny', 'niecelny', 'celny', 'niecelny', 'celny', 'out','celny', 'celny','out', 'niecelny', 'niecelny', 'niecelny', 'celny', 'celny', 'niecelny', 'niecelny', 'niecelny', 'celny', 'celny', 'niecelny', 'celny', 'celny', 'niecelny','celny','celny','celny', 'celny',  'celny', 'out'],
    'camera_under': 'under',
    'camera_site': 'site',
    'shooter': 'kuba',
    'last_clip_number': 1,
    'destination_path': output_dir
}
number_of_clips = len(video['end_time']) - 1
assert len(video['end_time']) == len(video['target']), 'To much target or times'
print((video['end_time'][54]))
print((video['target'][45:]))

for i, target in enumerate(video['target']):
    if (target != 'out'):

        ffmpeg_extract_subclip(video['path_site'], video['end_time'][i], video['end_time'][i+1], 
            targetname= output_dir /'site'/ f'{str(video["last_clip_number"])}_{video["shooter"]}_{video["camera_site"]}_{video["target"][i]}.mov')
        ffmpeg_extract_subclip(video['path_under'], video['end_time'][i], video['end_time'][i+1], 
            targetname= output_dir /'under'/ f'{str(video["last_clip_number"])}_{video["shooter"]}_{video["camera_under"]}_{video["target"][i]}.mov')
        video['last_clip_number'] += 1



