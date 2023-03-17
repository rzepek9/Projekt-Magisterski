from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# endtime poszczegolne czasy do wyciecia przykladowo start 0s koniec 7s, nastepne okno do wyciecia bedzie od 7s do nastepnego czasu
# last clip number oznacza ostatni wyciety w poprzednim wideo clip
#  nazwa clipu video['destination_path']+f'{video["shooter"]}_{video["camera"]}_{str(video["last_clip_number"])}_{video["target"][i]}.mov'
video = {
    'path': '/Users/jakubrzepkowski/Desktop/Studia/Praca Magisterska/Dataset/videos/kosz114.avi',
    'end_time': [0, 3*60+39, 3*60+47, 3*60+52, 3*60+56, 4*60+1, 4*60+6, 240+11, 240+16 , 240+21, 240+26, 240+31, 240+36, 240+42, 240+47, 240+53, 240+59, 300+4, 300+9, 300+16, 300+20, 300+26, 300+31, 300+36, 420, 420+4, 420+9, 420+15, 420+19, 420+23, 420+28, 420+33, 420+39, 420+44, 420+49,420+53, 420+57, 480+3,  480+7, 480+56, 540+2,540+6,540+11,540+16, 540+21, 540+25, 540+31, 540+36, 540+42, 540+48, 540+53, 540+58, 604, 613, 616, 621, 627, 632,637],
    'target': ['out', 'celny', 'celny', 'celny', 'celny',  'celny',  'celny',  'celny',  'celny',  'celny',  'celny',  'celny',  'celny',  'celny', 'celny', 'celny', 'celny', 'celny', 'celny', 'celny', 'celny', 'celny', 'celny', 'out', 'celny', 'celny',  'celny', 'celny', 'celny', 'celny', 'celny', 'celny', 'celny', 'celny', 'celny', 'out','celny', 'celny','out', 'celny', 'celny', 'celny', 'celny', 'celny', 'celny', 'celny', 'celny', 'celny', 'celny', 'celny', 'celny', 'celny', 'celny','celny','celny','celny', 'celny',  'celny','celny', 'out'],
    'camera': 'under',
    'shooter': 'kuba',
    'last_clip_number': 0,
    'destination_path': '/Users/jakubrzepkowski/Desktop/Studia/Praca Magisterska/Dataset/clip/ '
}
number_of_clips = len(video['end_time']) - 1

for i, target in enumerate(video['target']):
    if (target != 'out') & (i != number_of_clips):
        print(video['last_clip_number'])
        print(target)
        print(video['end_time'][i])
        ffmpeg_extract_subclip(video['path'], video['end_time'][i], video['end_time'][i+1], 
            targetname= video['destination_path']+f'{video["shooter"]}_{video["camera"]}_{str(video["last_clip_number"])}_{video["target"][i]}.mov')
        video['last_clip_number'] += 1



