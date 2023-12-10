import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from ball_trajectory_extraction.utils import (
    compute_angles,
    extract_ball_trajectory,
    interpolate_positions,
    normalize_positions,
    validate_trajectories,
    visualize_trajectory,
)
from basketball_detection.DetrForBasketballDetection import DetrForBasketballDetection
from pose_estimation.YoloPose import YoloPose
from shooting_motion_extraction import ShootingMotionExtractor
from tqdm import tqdm

INPUT_FILE = "C:/Users/gustaw/studia/magisterka_filmy/json_data/merged_trajectory.json"
OUTPUT_PATH = Path("C:/Users/gustaw/studia/magisterka_filmy/ball_trajectory")
PLOTS_PATH = OUTPUT_PATH / "trajectories_plots"
HALF_PLOTS_PATH = OUTPUT_PATH / "half_trajectories_plots"
ABOVE_HEAD_PLOTS_PATH = OUTPUT_PATH / "above_head_trajectories_plots_quadratic"
ABOVE_HEAD_PLOTS_HALF_PATH = OUTPUT_PATH / "above_head_half_trajectories_plots_quadratic"
READY_JSON = "C:/Users/gustaw/studia/magisterka_filmy/ball_trajectory/extracted_trajectories_quadratic.json"
OUTPUT_JSON = "C:/Users/gustaw/studia/magisterka_filmy/ball_trajectory/extracted_trajectories_final.json"
FAILED_FILES_TXT = "C:/Users/gustaw/studia/magisterka_filmy/ball_trajectory/failed_files.txt"
MAPPING_PATH = "C:/Users/gustaw/studia/magisterka_filmy/3d_cnn_data/gustaw_mapping.csv"
DATABASE_PATH = "C:/Users/gustaw/studia/magisterka_filmy/3d_cnn_data/database.csv"

BBALL_DETR_CHECKPOINT = "basketball_detection/detr-bball_101"
DETR_101 = "facebook/detr-resnet-101"

yolo_pose = YoloPose()
bball_detector = DetrForBasketballDetection(BBALL_DETR_CHECKPOINT, DETR_101)
shooting_motion_extractor_detr = ShootingMotionExtractor(yolo_pose, bball_detector)


def remove_extension(filename):
    if ".avi" in filename:
        return filename.replace(".avi", "")
    elif ".mov" in filename:
        return filename.replace(".mov", "")
    else:
        return filename


def get_mapping(mapping_path):
    mapping = pd.read_csv(mapping_path, index_col=0)
    mapping = mapping.drop(columns=["under_old", "under_new"])
    columns_to_modify = ["side_old", "side_new"]
    for col in columns_to_modify:
        mapping[col] = mapping[col].apply(remove_extension)
    return mapping


def get_database(database_path):
    database = pd.read_csv(database_path)
    database = database[~database["file"].str.contains("under")]
    return database


def extract_filename_from_path(filepath):
    """Extracts the filename without extension from a given filepath."""
    return os.path.splitext(os.path.basename(filepath))[0]


def get_new_filename(old_filename, mapping_df):
    """Maps the old filename to the new filename using the provided mapping dataframe."""
    new_filename = mapping_df[mapping_df["side_old"] == old_filename]["side_new"].values
    return new_filename[0] if new_filename.size > 0 else old_filename


def fetch_labels(new_filename, database_df):
    """Fetches the labels for the given filename from the database dataframe."""
    labels = database_df[database_df["file"] == new_filename][["made", "clean", "skill"]]
    return labels.iloc[0].to_dict() if not labels.empty else None


def process_labels(filepath, mapping_df, database_df):
    """Processes the trajectory data to prepare it for neural network training."""
    old_filename = extract_filename_from_path(filepath)
    new_filename = get_new_filename(old_filename, mapping_df)
    labels = fetch_labels(new_filename, database_df)
    return labels


def visualize_and_save(trajectories, basket, filename):
    output_file = PLOTS_PATH / f"{Path(filename).stem}.png"
    visualize_trajectory(basket, trajectories, filename=output_file)


def visualize_and_save_half(trajectories, basket, filename):
    output_file = HALF_PLOTS_PATH / f"{Path(filename).stem}.png"
    visualize_trajectory(basket, trajectories, filename=output_file)


def visualize_and_save_above_head(trajectories, basket, filename):
    output_file = ABOVE_HEAD_PLOTS_PATH / f"{Path(filename).stem}.png"
    visualize_trajectory(basket, trajectories, filename=output_file)


def visualize_and_save_above_head_half(trajectories, basket, filename):
    output_file = ABOVE_HEAD_PLOTS_HALF_PATH / f"{Path(filename).stem}.png"
    visualize_trajectory(basket, trajectories, filename=output_file)


def convert_numpy_to_list(data):
    for frame_data in data:
        for key, value in frame_data.items():
            if isinstance(value, np.ndarray):
                frame_data[key] = value.tolist()
    return data


if __name__ == "__main__":
    mapping_df = get_mapping(MAPPING_PATH)
    database_df = get_database(DATABASE_PATH)
    failed_files = []
    extracted_data = {}

    # Ensure the plots directory exists
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    HALF_PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    ABOVE_HEAD_PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    ABOVE_HEAD_PLOTS_HALF_PATH.mkdir(parents=True, exist_ok=True)

    with open(INPUT_FILE) as f:
        data = json.load(f)

    for filename, frames_data in tqdm(data.items(), "Extracting trajectories"):
        (
            trajectories,
            first_half_trajectories,
            above_head_trajectories,
            above_head_half_trajectories,
            basket,
            ball_release_frame,
            shot_flight_end_frame,
            top_frame,
        ) = extract_ball_trajectory(frames_data)
        if trajectories is not None and validate_trajectories(trajectories, top_frame):
            visualize_and_save(trajectories, basket, filename)
            visualize_and_save_half(first_half_trajectories, basket, filename)
            visualize_and_save_above_head(above_head_trajectories, basket, filename)
            visualize_and_save_above_head_half(above_head_half_trajectories, basket, filename)
            trajectories = interpolate_positions(trajectories)
            first_half_trajectories = interpolate_positions(first_half_trajectories)
            above_head_trajectories = interpolate_positions(above_head_trajectories)
            above_head_half_trajectories = interpolate_positions(above_head_half_trajectories)
            normalized_trajectories = normalize_positions(trajectories, basket)
            normalized_first_half_trajectories = normalize_positions(first_half_trajectories, basket)
            normalized_above_head_trajectories = normalize_positions(above_head_trajectories, basket)
            normalized_above_head_half_trajectories = normalize_positions(
                above_head_half_trajectories, basket
            )
            angles = compute_angles(normalized_trajectories)
            first_half_angles = compute_angles(normalized_first_half_trajectories)
            above_head_angles = compute_angles(normalized_above_head_trajectories)
            above_head_half_angles = compute_angles(normalized_above_head_half_trajectories)
            extracted_data[filename] = {
                "trajectories": trajectories,
                "first_half_trajectories": first_half_trajectories,
                "above_head_trajectories": above_head_trajectories,
                "above_head_half_trajectories": above_head_half_trajectories,
                "normalized_trajectories": normalized_trajectories,
                "normalized_first_half_trajectories": normalized_first_half_trajectories,
                "normalized_above_head_trajectories": normalized_above_head_trajectories,
                "normalized_above_head_half_trajectories": normalized_above_head_half_trajectories,
                "angles": angles,
                "first_half_angles": first_half_angles,
                "above_head_angles": above_head_angles,
                "above_head_half_angles": above_head_half_angles,
                "basket": basket,
                "ball_release_frame": ball_release_frame,
                "shot_flight_end_frame": shot_flight_end_frame,
                "top_frame_index": top_frame,
            }
            labels = process_labels(filename, mapping_df, database_df)
            if labels:
                extracted_data[filename].update(labels)
        else:
            failed_files.append(filename)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(extracted_data, f)

    # Print the number of failed files and save them to a .txt file
    print(f"Number of files initially not correctly processed: {len(failed_files)}")
    with open(FAILED_FILES_TXT, "w") as f:
        for file in failed_files:
            f.write(f"{file}\n")

    with open(READY_JSON, "r") as f:
        extracted_data = json.load(f)

    with open(FAILED_FILES_TXT, "r") as f:
        failed_files = [line.strip() for line in f.readlines()]

    # retry loop with ball_confidence 0.75
    remaining_failed_files = []
    gucio_files = [file for file in failed_files if "gustaw" in file]
    for failed_file in tqdm(gucio_files, "Retrying failed files with ball_confidence 0.75"):
        collected_data = shooting_motion_extractor_detr.collect_data_from_video(
            str(failed_file), ball_confidence=0.75, perspective="side"
        )
        collected_data = convert_numpy_to_list(collected_data)
        (
            trajectories,
            first_half_trajectories,
            above_head_trajectories,
            above_head_half_trajectories,
            basket,
            ball_release_frame,
            shot_flight_end_frame,
            top_frame,
        ) = extract_ball_trajectory(collected_data)
        if trajectories is None or not validate_trajectories(trajectories, top_frame):
            remaining_failed_files.append(failed_file)
        if trajectories is not None and validate_trajectories(trajectories, top_frame):
            visualize_and_save(trajectories, basket, failed_file)
            visualize_and_save_half(first_half_trajectories, basket, failed_file)
            visualize_and_save_above_head(above_head_trajectories, basket, failed_file)
            visualize_and_save_above_head_half(above_head_half_trajectories, basket, failed_file)
            trajectories = interpolate_positions(trajectories)
            first_half_trajectories = interpolate_positions(first_half_trajectories)
            above_head_trajectories = interpolate_positions(above_head_trajectories)
            above_head_half_trajectories = interpolate_positions(above_head_half_trajectories)
            normalized_trajectories = normalize_positions(trajectories, basket)
            normalized_first_half_trajectories = normalize_positions(first_half_trajectories, basket)
            normalized_above_head_trajectories = normalize_positions(above_head_trajectories, basket)
            normalized_above_head_half_trajectories = normalize_positions(
                above_head_half_trajectories, basket
            )
            angles = compute_angles(normalized_trajectories)
            first_half_angles = compute_angles(normalized_first_half_trajectories)
            above_head_angles = compute_angles(normalized_above_head_trajectories)
            above_head_half_angles = compute_angles(normalized_above_head_half_trajectories)
            extracted_data[failed_file] = {
                "trajectories": trajectories,
                "first_half_trajectories": first_half_trajectories,
                "above_head_trajectories": above_head_trajectories,
                "above_head_half_trajectories": above_head_half_trajectories,
                "normalized_trajectories": normalized_trajectories,
                "normalized_first_half_trajectories": normalized_first_half_trajectories,
                "normalized_above_head_trajectories": normalized_above_head_trajectories,
                "normalized_above_head_half_trajectories": normalized_above_head_half_trajectories,
                "angles": angles,
                "first_half_angles": first_half_angles,
                "above_head_angles": above_head_angles,
                "above_head_half_angles": above_head_half_angles,
                "basket": basket,
                "ball_release_frame": ball_release_frame,
                "shot_flight_end_frame": shot_flight_end_frame,
                "top_frame_index": top_frame,
            }
            labels = process_labels(failed_file, mapping_df, database_df)
            if labels:
                extracted_data[failed_file].update(labels)
            print(f"Successfully processed {failed_file} with ball_confidence 0.7")
        if trajectories is None or not validate_trajectories(trajectories, top_frame):
            print(f"Failed to process {failed_file} with ball_confidence 0.7")
            remaining_failed_files.append(failed_file)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(extracted_data, f)

    # Print the number of failed files and save them to a .txt file
    print(f"Number of files not correctly processed: {len(remaining_failed_files)}")
    with open(FAILED_FILES_TXT, "w") as f:
        for file in remaining_failed_files:
            f.write(f"{file}\n")
