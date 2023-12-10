from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pathlib import Path
import os

# endtime poszczegolne czasy do wyciecia przykladowo start 0s koniec 7s, nastepne okno do wyciecia bedzie od 7s do nastepnego czasu
# last clip number oznacza ostatni wyciety w poprzednim wideo clip
#  nazwa clipu video['destination_path']+f'{video["shooter"]}_{video["camera"]}_{str(video["last_clip_number"])}_{video["target"][i]}.mov'

input_dir = Path('/Users/jakubrzepkowski/Documents/Projekt-magisterski/Wideo/KOSZ')
output_dir = Path('/Users/jakubrzepkowski/Documents/Projekt-magisterski/Clipy_wideo')

clip_site_dir = Path('/Users/jakubrzepkowski/Documents/Projekt-magisterski/Clipy_wideo/site')
clip_under_dir = Path('/Users/jakubrzepkowski/Documents/Projekt-magisterski/Clipy_wideo/under')
clip_4_second_dir = Path('/Users/jakubrzepkowski/Documents/Projekt-magisterski/four_seconds')

def crop_clips():
    video = {
        'path_under': str(input_dir /'pierwsze_nagrania.avi'),
        'path_site': str(input_dir /'webcam3.avi'),
        'end_time': 
        # [(9.5, 'celny'), (13.5, 'celny'), (18, 'celny'), (23, 'niecelny'), (28, 'niecelny'), (32, 'celny'), (37, 'niedolot'),
        #             (42, 'celny'), (47, 'niecelny'), (43, 'celny'),(48, 'niecelny'), (52, 'niecelny'), (56, 'niecelny'), (61, 'celny'), (66, 'niecelny'), 
        #             (70, 'out'), (98, 'celny'), (1*60+43, 'celny'), (107, 'out'), (112, 'celny'), (117, 'celny'), (122, 'celny'), (127, 'niecelny'), 
        #             (131, 'celny'), (136, 'celny'), (140, 'celny'), (145, 'celny'), (149, 'out'), (3*60+2, 'celny'), (3*60+7, 'celny'), 
        #             (3*60+12, 'niecelny'), (3*60+16, 'celny'), (3*60+21, 'celny'), (3*60+25, 'niecelny'), (3*60+31, 'celny'), 
        #             (3*60+36, 'niecelny'), (3*60+40.4, 'niecelny'), (3*60+46, 'celny'), (3*60+50, 'niecelny'), (3*60+54, 'out')], 
                    # [(4*60+25, 'celny'),
                    # (4*60+31, 'celny'), (4*60+35, 'niecelny'), (4*60+40, 'celny'), (4*60+45, 'celny'), (4*60+52, 'celny'), (4*60+56.5, 'niecelny'), 
                    # (5*60+2, 'niecelny'), (5*60+7, 'niecelny'), (5*60+12, 'celny'), (5*60+19, 'celny'), (5*60+24, 'niecelny'), (5*60+32, 'celny'), 
                    # (5*60+37, 'nicelny'),(5*60+41, 'out'), (6*60+11.5, 'celny'), (6*60+17, 'celny'), (6*60+23, 'celny'), (6*60+28, 'celny'), 
                    # (6*60+33.5, 'niecelny'), (6*60+41, 'celny'), (6*60+47, 'celny'), (6*60+53, 'celny'), (6*60+59, 'celny'), (7*60+4.3, 'celny'), 
                    # (7*60+11, 'celny'), (7*60+16, 'celny'), (7*60+21, 'out'), (7*60+25, 'out'), (8*60+2, 'niecelny'), (8*60+8, 'celny'), 
                    # (8*60+14, 'celny'), (8*60+19, 'celny'), (8*60+25, 'niecelny'), (8*60+30, 'celny'), 
                    # (8*60+36, 'niecelny'), (8*60+42, 'celny'), (8*60+47, 'celny'), (8*60+52, 'niecelny '),(9*60, 'niecelny'), (9*60+6, 'niecelny'),
                    # (9*60+13, 'niecelny'), (9*60+17, 'out')],
                    # [(9*60+45, 'niecelny'),
                    # (9*60+49, 'out'), (10*60, 'celny'), (10*60+4.7, 'niecelny'), (10*60+11, 'celny'), (10*60+16, 'celny'), (10*60+21, 'celny'), 
                    # (10*60+28, 'niecelny'), (10*60+33, 'niecelny'), (10*60+38.4, 'celny'), (10*60+44.8, 'niecelny'), (10*60+50, 'niedolot'), (10*60+55, 'celny'), 
                    # (11*60, 'niecelny'),(11*60+6, 'niecelny'), (11*60+11, 'out'), (11*60+37, 'niecelny'), (11*60+43, 'celny'), (11*60+48, 'celny'), 
                    # (11*60+54, 'celny'), (12*60, 'celny'), (12*60+6, 'celny'), (12*60+12, 'niecelny'), (12*60+18, 'celny'), (12*60+24.2, 'celny'), 
                    # (12*60+30, 'celny'), (12*60+34, 'out'), (13*60+8.6, 'niecelny'), (13*60+15, 'celny'), (13*60+21, 'niecelny'), (13*60+27.4, 'celny'), 
                    # (13*60+34, 'niecelny'), (13*60+38, 'out'), (13*60+46, 'celny'), (13*60+51.4, 'celny'), 
                    # (13*60+57.3, 'celny'), (14*60+4, 'niedolot'), (14*60+9.5, 'niedolot'), (14*60+18, 'celny '),(14*60+23.8, 'niecelny'), (14*60+29, 'niecelny'),
                    # (14*60+33, 'out'),],
                    [(16*60+1.5, 'niecelny'),
                    (16*60+7, 'celny'), (16*60+11.3, 'niecelny'), (16*60+17, 'niecelny'), (16*60+21, 'celny'), (16*60+26.5, 'niecelny'), (16*60+32, 'celny'), 
                    (16*60+36.2, 'niedolot'), (10*60+40.8, 'celny'), (10*60+45.5, 'celny'), (10*60+50, 'out')],

        'camera_under': 'under',
        'camera_site': 'site',
        'shooter': 'kuba',
        'last_clip_number': 240,
        'destination_path': output_dir
    }
    number_of_clips = len(video['end_time']) - 1
    # assert len(video['end_time']) == len(video['target']), 'To much target or times'
    # print((video['end_time'][54]))
    # print((video['target'][45:]))

    for i, target in enumerate(video['end_time']):
        if (target[1] != 'out'):
            video['last_clip_number'] += 1
            if video['last_clip_number'] in [139,138]:
                ffmpeg_extract_subclip(video['path_site'], video['end_time'][i][0]-1, video['end_time'][i+1][0]-1, 
                    targetname= output_dir /'site'/ f'{str(video["last_clip_number"])}_{video["shooter"]}_{video["camera_site"]}_{target[1]}.mov')
                ffmpeg_extract_subclip(video['path_under'], video['end_time'][i][0]-1, video['end_time'][i+1][0]-1, 
                    targetname= output_dir /'under'/ f'{str(video["last_clip_number"])}_{video["shooter"]}_{video["camera_under"]}_{target[1]}.mov')
            else:
                ffmpeg_extract_subclip(video['path_site'], video['end_time'][i][0], video['end_time'][i+1][0], 
                    targetname= output_dir /'site'/ f'{str(video["last_clip_number"])}_{video["shooter"]}_{video["camera_site"]}_{target[1]}.mov')
                ffmpeg_extract_subclip(video['path_under'], video['end_time'][i][0], video['end_time'][i+1][0], 
                    targetname= output_dir /'under'/ f'{str(video["last_clip_number"])}_{video["shooter"]}_{video["camera_under"]}_{target[1]}.mov')



def crop_on_4_seconds_clip(files_dir, camera):
    list_1_second = [1, 5, 21, 22, 36, 50, 86, 88, 89, 90, 91, 95, 96, 97, 98, 99, 100, 101, 105, 106, 107, 108, 109, 110, 118]
    list_2_second = [44, 74, 14]
    list_0_8_second = [116, 71]
    
    file_clip_list = []

    for file in os.listdir(files_dir):
        if file.endswith(".mov"):
            file_clip_list.append(file)
   
    for f in file_clip_list:
        number = f.split('_', 1)[0]
        print(f'{number}; {f};')
        if int(number) in list_1_second:
            ffmpeg_extract_subclip(files_dir / f, 1, 5, 
                targetname= clip_4_second_dir / camera / f)
        
        elif int(number) in list_2_second:
            ffmpeg_extract_subclip(files_dir / f, 2, 6, 
                targetname= clip_4_second_dir / camera / f)
        
        elif int(number) in list_0_8_second:
            ffmpeg_extract_subclip(files_dir / f, 0.8, 4.8, 
                targetname= clip_4_second_dir / camera / f)
        
        else:
            ffmpeg_extract_subclip(files_dir / f, 0, 4, 
                targetname= clip_4_second_dir / camera / f)
        


            
if __name__ == '__main__':
    crop_on_4_seconds_clip(clip_under_dir, 'under')