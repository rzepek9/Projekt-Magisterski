from pathlib import Path

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

from shooting_motion_extraction.ShootingMotionExtractor import ShootingMotionExtractor

OUTPUT_PATH = Path("/home/gustaw/magisterka/videos_extracted/")
INPUT_DIR = Path("/home/gustaw/magisterka/four_seconds/")

shooting_motion_extractor = ShootingMotionExtractor()


def get_output_filename(source):
    source_name = source.name
    video_type = source.parent.name
    if not OUTPUT_PATH.exists():
        OUTPUT_PATH.mkdir(parents=True)
        print(f"Created {OUTPUT_PATH} directory")
    output_dir = OUTPUT_PATH.joinpath(video_type)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print(f"Created {output_dir} directory")
    return str(output_dir.joinpath(source_name))


def get_shooting_motion_period(source):
    return shooting_motion_extractor.extract_shooting_motion(
        str(source), show_video=False, return_shooting_motion_period=True
    )


def crop_video(source_under, source_side):
    start_time, end_time = get_shooting_motion_period(source_under)
    if not (start_time and end_time):
        print("RETRYING FROM THE SIDE PERSPECTIVE")
        start_time, end_time = get_shooting_motion_period(source_side)
    if start_time and end_time:
        output_filename_under = get_output_filename(source_under)
        output_filename_side = get_output_filename(source_side)
        ffmpeg_extract_subclip(
            source_under, start_time, end_time, targetname=output_filename_under
        )
        ffmpeg_extract_subclip(
            source_side, start_time, end_time, targetname=output_filename_side
        )
    else:
        print("Unable to extract the shooting motion")
        source_under_failed = source_under.with_suffix(".avi").with_name(
            source_under.stem + "_FAILED.mov"
        )
        source_side_failed = source_side.with_suffix(".avi").with_name(
            source_side.stem + "_FAILED.mov"
        )
        source_under.rename(source_under_failed)
        source_side.rename(source_side_failed)
        pass


if __name__ == "__main__":
    under_dir = INPUT_DIR.joinpath("under")
    side_dir = INPUT_DIR.joinpath("side")
    files = [
        (
            under_file,
            side_dir.joinpath(under_file.name.replace("_under_", "_side_")),
        )
        for under_file in under_dir.iterdir()
        if under_file.name.endswith(".mov")
    ]
    print(len(files))
    exists = 0
    for under_file, side_file in files:
        """if both output files for under and side extst exist += 1 etc.
        Write if *side output file exists* and *under output file exists*:"""
        if Path(get_output_filename(side_file)).exists() and Path(get_output_filename(under_file)).exists():
            exists += 1
            print(f"ALREADY EXTRACTED {exists} FILES")
        else:
            if side_file.exists():
                print(f"PROCESSING{under_file}")
                crop_video(under_file, side_file)
            else:
                print(f"No matching file for {under_file} found")
