import math

BBALL_SIZE_THRESHOLD = 650
SCALE_FACTOR = 480 / 512


def is_ball_too_big(ball_coords: list) -> bool:
    """
    Checks if ball is above the threshold size
    """
    return get_ball_size(ball_coords) > BBALL_SIZE_THRESHOLD


def get_hand_ball_distances(
    ball_coords: list, left_wrist_coords: list, right_wrist_coords: list
) -> tuple:
    """
    Counts the distances from hands to the ball center
    Returns the tuple with the distances
    """
    ball_center = (
        (ball_coords[0] + ball_coords[2]) / 2,
        (ball_coords[1] + ball_coords[3]) / 2,
    )
    left_hand_dist = get_distance(ball_center, left_wrist_coords)
    right_hand_dist = get_distance(ball_center, right_wrist_coords)
    return left_hand_dist, right_hand_dist


def get_elbow_ball_distances(
    ball_coords: list, left_elbow_coords: list, right_elbow_coords: list
) -> tuple:
    """
    Counts the distances from elbows to the ball center
    Returns the tuple with the distances
    """
    ball_center = (
        ball_coords[0] + ball_coords[2] / 2,
        ball_coords[1] + ball_coords[3] / 2,
    )
    left_elbow_dist = get_distance(ball_center, left_elbow_coords)
    right_elbow_dist = get_distance(ball_center, right_elbow_coords)
    return left_elbow_dist, right_elbow_dist


def get_distance(ball_center: list, kpt_coords: list) -> float:
    """
    Counts and returns the distance between ball center and given coordinates
    """
    return math.sqrt(
        (ball_center[0] - kpt_coords[0]) ** 2 + (ball_center[1] - kpt_coords[1]) ** 2
    )


def init_extraction_vars() -> tuple:
    """
    Initializes variables for the extraction of shooting motion
    """
    return None, 0, None, None


def get_distances(ball_coords: list, kpts_without_head: list) -> tuple:
    """ """
    hands = kpts_without_head[6:18]
    left_elbow = rescale_detections(hands[:2])
    right_elbow = rescale_detections(hands[3:5])
    left_wrist = rescale_detections(hands[6:8])
    right_wrist = rescale_detections(hands[9:11])
    left_hand_dist, right_hand_dist = get_hand_ball_distances(
        ball_coords, left_wrist, right_wrist
    )
    left_elbow_dist, right_elbow_dist = get_elbow_ball_distances(
        ball_coords, left_elbow, right_elbow
    )
    return left_hand_dist, right_hand_dist, left_elbow_dist, right_elbow_dist


def rescale_detections(detection_coords: list, scale_factor=None) -> list:
    """
    Rescales the detections based on a given scale factor
    Needed because of different frame sizes for bball and yolopose detections
    """
    if not scale_factor:
        scale_factor = SCALE_FACTOR
    detection_coords[1] *= scale_factor
    return detection_coords


def get_ball_size(ball_coords) -> int:
    """ "
    Counts and returns the ball size based on its coordinates
    """
    return (ball_coords[2] - ball_coords[0]) * (ball_coords[3] - ball_coords[1])
