import json
from pathlib import Path

import numpy as np
from ball_trajectory_extraction import ShootingDataExtractor
from basketball_detection import DetrForBasketballDetection
from pose_estimation import YoloPose
from tqdm import tqdm


def load_partial_data(folder_name, json_data_dir):
    partial_file_path = json_data_dir / f"{folder_name}_partial.json"
    if partial_file_path.exists():
        with open(partial_file_path, "r") as f:
            return json.load(f)
    return {}


def collect_data_for_folder(folder, extractor, json_data_dir, ball_confidence=0.75, save_interval=100):
    folder_name = folder[0].parent.name if folder else "Unknown Folder"
    counter = 0

    # Load the partial data
    partial_file_path = f"{str(json_data_dir)}/{folder_name}_partial.json"
    if Path(partial_file_path).exists():
        with open(partial_file_path, "r") as f:
            partial_data = json.load(f)
    else:
        partial_data = {}

    # Initialize the main data with the partial data
    data = partial_data

    for file in tqdm(folder, desc=f"Processing files in {folder_name}"):
        # Check if the file is already in the partial data
        if str(file) in partial_data:
            continue

        try:
            result = extractor.collect_data_from_video(
                str(file),
                ball_confidence=ball_confidence,
            )

            # Convert numpy arrays to lists
            for frame_data in result:
                for key, value in frame_data.items():
                    if isinstance(value, np.ndarray):
                        frame_data[key] = value.tolist()
            data[str(file)] = result
            counter += 1

            # Save data periodically
            if counter % save_interval == 0:
                with open(partial_file_path, "w") as f:
                    json.dump(data, f)
                print(f"Saved partial data for {folder_name} after processing {counter} videos.")

        except Exception as e:
            print(f"Error processing {file}: {e}")
            data[str(file)] = {"error": str(e)}

    return data


if __name__ == "__main__":
    folder_side_gucio = Path("D:/side_ok")
    folder_side_kuba = Path("D:/side_kuba")
    folder_under_gucio = Path("D:/under_ok")
    folder_under_kuba = Path("D:/under_kuba")

    side_gucio = [file for file in folder_side_gucio.glob("*.avi") if file.is_file()]
    side_kuba = [file for file in folder_side_kuba.glob("*.mov") if file.is_file()]
    under_gucio = [file for file in folder_under_gucio.glob("*.avi") if file.is_file()]
    under_kuba = [file for file in folder_under_kuba.glob("*.mov") if file.is_file()]

    folders = [side_gucio, side_kuba, under_gucio, under_kuba]
    folders_side = [side_gucio, side_kuba]
    folders_under = [under_gucio, under_kuba]

    folders = [side_gucio, side_kuba, under_gucio, under_kuba]
    folders_side = [side_gucio, side_kuba]
    folders_under = [under_gucio, under_kuba]

    # Define output directories
    output_dir_side = Path("C:/Users/gustaw/studia/magisterka_filmy/side_processed_reprocessed")
    output_dir_under = Path("C:/Users/gustaw/studia/magisterka_filmy/under_processed")
    json_data_dir = Path("C:/Users/gustaw/studia/magisterka_filmy/json_data")

    # Create directories if they don't exist
    output_dir_side.mkdir(parents=True, exist_ok=True)
    output_dir_under.mkdir(parents=True, exist_ok=True)
    json_data_dir.mkdir(parents=True, exist_ok=True)
    yolo_pose = YoloPose()

    detr_detector = DetrForBasketballDetection(
        "basketball_detection/detr-bball_101", "facebook/detr-resnet-101"
    )

    shooting_motion_extractor_detr = ShootingDataExtractor(yolo_pose, detr_detector)

    # Process side_gucio folder
    side_data_detr_gucio = collect_data_for_folder(
        side_gucio, shooting_motion_extractor_detr, json_data_dir, ball_confidence=0.75
    )
    with open(f"{str(json_data_dir)}/side_gucio_trajectory.json", "w") as f:
        json.dump(side_data_detr_gucio, f)

    # Process side_kuba folder
    side_data_detr_kuba = collect_data_for_folder(
        side_kuba, shooting_motion_extractor_detr, json_data_dir, ball_confidence=0.75
    )
    with open(f"{str(json_data_dir)}/side_data_detr_kuba.json", "w") as f:
        json.dump(side_data_detr_kuba, f)
