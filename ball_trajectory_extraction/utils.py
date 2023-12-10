import math

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def find_basket(frames_data):
    basket_detections = [frames_data[i]["baskets"] for i in range(len(frames_data))]
    basket_detections = [basket for basket in basket_detections if basket is not None]
    final_basket = None
    for baskets in basket_detections:
        if final_basket is None:
            final_basket = sorted(baskets, key=lambda x: x[0])[0]
        else:
            final_basket = (
                sorted(baskets, key=lambda x: x[0])[0]
                if final_basket[0] > sorted(baskets, key=lambda x: x[0])[0][0]
                else final_basket
            )
    if final_basket[1] > 600:
        final_basket = None
    return final_basket


def get_iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0


def is_ball_above_basket(ball, basket):
    avg_ball_y = (ball[1] + ball[3]) / 2
    avg_basket_y = (basket[1] + basket[3]) / 2
    if avg_ball_y < avg_basket_y - 100:
        return True
    else:
        return False


def is_ball_to_right_of_basket(ball, basket):
    avg_ball_x = (ball[0] + ball[2]) / 2
    avg_basket_x = (basket[0] + basket[2]) / 2
    if avg_ball_x > avg_basket_x + 100:
        return True
    else:
        return False


def get_person_bbox_from_frame(frame_data):
    frame_person = frame_data.get("person_bbox")
    if frame_person is None:
        return None
    else:
        return frame_person[0]


def is_ball_to_left_of_person(ball, person_bbox, postprocessing=False):
    avg_ball_x = (ball[0] + ball[2]) / 2
    if person_bbox is None:
        return False
    avg_person_x = (person_bbox[0] + person_bbox[2]) / 2
    if postprocessing:
        avg_person_x -= 50
    if avg_ball_x < avg_person_x:
        return True
    else:
        return False


def get_person_keypoints_from_frame(frame_data):
    frame_person = frame_data.get("keypoints")
    if frame_person is None:
        return None
    else:
        return frame_person[0]


def validate_highest_ball(ball, basket, person_bbox):
    if (
        is_ball_above_basket(ball, basket)
        and is_ball_to_right_of_basket(ball, basket)
        and is_ball_to_left_of_person(ball, person_bbox)
    ):
        return True
    else:
        return False


def find_highest_ball_position(frames_data):
    basket = find_basket(frames_data)
    if basket is None:
        return None, None
    if len(basket) == 0:
        return None, None
    n_frames = len(frames_data)
    ball_detections = [frames_data[i]["balls"] for i in range(len(frames_data))]
    ball_detections = [(frame, ball) for frame, ball in enumerate(ball_detections) if ball is not None]
    highest_ball = None
    highest_ball_frame = None
    for frame, balls in ball_detections:
        if frame >= n_frames - 10 or frame <= 10:
            continue
        person_bbox = get_person_bbox_from_frame(frames_data[frame])
        if person_bbox is not None and len(balls) > 0:
            for ball in balls:
                if validate_highest_ball(ball, basket, person_bbox):
                    # Check if the ball is moving from right to left for 3 consecutive frames before and after
                    is_moving_right_to_left = True
                    for i in range(-3, 4):  # Check 3 frames before and 3 frames after
                        if frame + i < 0 or frame + i >= n_frames:
                            continue
                        current_ball = get_top_ball_from_frame(frames_data[frame + i], basket)
                        next_ball = (
                            get_top_ball_from_frame(frames_data[frame + i + 1], basket)
                            if frame + i + 1 < n_frames
                            else None
                        )
                        if (
                            current_ball is not None
                            and next_ball is not None
                            and current_ball[0] <= next_ball[0]
                        ):  # Ball is moving left to right
                            is_moving_right_to_left = False
                            break
                    if not is_moving_right_to_left:
                        continue
                    if highest_ball is None and highest_ball_frame is None:
                        highest_ball = ball
                        highest_ball_frame = frame
                    else:
                        if ball[1] < highest_ball[1] and validate_highest_ball(ball, basket, person_bbox):
                            highest_ball = ball
                            highest_ball_frame = frame
    return highest_ball, highest_ball_frame


def is_ball_above_shoulders(ball, person_keypoints):
    if len(person_keypoints) >= 7:
        left_shoulder_y = person_keypoints[5][1]
        right_shoulder_y = person_keypoints[6][1]
        shoulder_avg_y = (left_shoulder_y + right_shoulder_y) / 2
        avg_ball_y = (ball[1] + ball[3]) / 2
        if avg_ball_y < shoulder_avg_y:
            return True
        else:
            return False
    else:
        return False


def is_ball_above_hips(ball, person_keypoints):
    if len(person_keypoints) >= 14:
        left_hip_y = person_keypoints[11][1]
        right_hip_y = person_keypoints[12][1]
        hip_avg_y = (left_hip_y + right_hip_y) / 2
        avg_ball_y = (ball[1] + ball[3]) / 2
        return avg_ball_y < hip_avg_y
    return False


def get_top_ball_from_frame(frame_data, basket):
    frame_balls = frame_data.get("balls")
    person_bbox = get_person_bbox_from_frame(frame_data)
    person_keypoints = get_person_keypoints_from_frame(frame_data)

    if frame_balls is None or person_bbox is None or person_keypoints is None:
        return None

    # Filter balls based on the conditions
    valid_balls = [
        ball
        for ball in frame_balls
        if is_ball_to_right_of_basket(ball, basket)
        and is_ball_above_hips(ball, person_keypoints)
        and (is_ball_to_left_of_person(ball, person_bbox) or get_iou(ball, person_bbox) > 0)
    ]

    if valid_balls is None:
        return None

    # Return the highest valid ball
    if len(valid_balls) == 0:
        return None
    return sorted(valid_balls, key=lambda x: x[1])[0]


def find_ball_release(frames_data, highest_ball_frame):
    release_frame = None
    basket = find_basket(frames_data)
    if basket is None:
        return None
    if len(basket) == 0:
        return None

    highest_ball, _ = find_highest_ball_position(frames_data)
    if highest_ball is None:
        return None

    avg_x_highest_ball = (highest_ball[0] + highest_ball[2]) / 2

    for frame in range(highest_ball_frame, -1, -1):
        frame_data = frames_data[frame]

        person_keypoints = get_person_keypoints_from_frame(frame_data)
        person_bbox = get_person_bbox_from_frame(frame_data)
        if person_keypoints is None or person_bbox is None:
            continue
        ball = get_top_ball_from_frame(frame_data, basket)

        if ball is not None:
            avg_x_ball = (ball[0] + ball[2]) / 2

            # Check if the ball is not more to the left than the highest ball
            if avg_x_ball > avg_x_highest_ball:
                if is_ball_above_shoulders(ball, person_keypoints):
                    continue
                else:
                    return frame + 1

    return release_frame


def find_shot_flight_end(frames_data, highest_ball_frame):
    basket = find_basket(frames_data)
    if basket is None:
        return None
    if len(basket) == 0:
        return None
    last_frame_above_basket = None
    prev_ball = None

    highest_ball, _ = find_highest_ball_position(frames_data)
    if highest_ball is None:
        return None

    avg_x_highest_ball = (highest_ball[0] + highest_ball[2]) / 2
    avg_x_basket = (basket[0] + basket[2]) / 2

    for frame in range(highest_ball_frame, len(frames_data)):
        frame_data = frames_data[frame]
        ball = get_top_ball_from_frame(frame_data, basket)

        if ball is not None:
            avg_x_ball = (ball[0] + ball[2]) / 2

            # Check if the ball is between the highest ball detected and the basket
            if avg_x_basket < avg_x_ball < avg_x_highest_ball:
                # Check if the ball is moving downwards and from right to left
                if prev_ball is not None and ball[1] > prev_ball[1] and ball[0] < prev_ball[0]:
                    if is_ball_above_basket(ball, basket):
                        last_frame_above_basket = frame
                    else:
                        return last_frame_above_basket

            prev_ball = ball

    return last_frame_above_basket


def post_process_trajectories(first_half_trajectories, trajectories):
    # Create new lists for post-processed trajectories
    new_first_half = []
    new_trajectories = []

    # Post-process the first half of the shot
    for i in range(len(first_half_trajectories) - 1):
        current_ball = first_half_trajectories[i]
        next_ball = first_half_trajectories[i + 1]

        if next_ball is not None and current_ball is not None:
            if next_ball[0] >= current_ball[0] and next_ball[1] <= current_ball[1]:
                new_first_half.append(current_ball)
        elif next_ball is None:
            new_first_half.append(None)

    # Ensure the last ball is added
    new_first_half.append(first_half_trajectories[-1])

    # Post-process the start of the first half of the shot
    while len(new_first_half) > 3 and all(x is None for x in new_first_half[:3]):
        new_first_half = new_first_half[3:]

    # Post-process the end of the first half of the shot
    while len(new_first_half) > 3 and all(x is None for x in new_first_half[-3:]):
        new_first_half = new_first_half[:-3]

    # Start the new_trajectories list with the post-processed first half
    new_trajectories.extend(new_first_half)

    # Post-process the second half of the shot
    for i in range(len(trajectories) - len(first_half_trajectories), len(trajectories) - 1):
        current_ball = trajectories[i]
        prev_ball = trajectories[i - 1]

        if prev_ball is not None and current_ball is not None:
            if prev_ball[0] >= current_ball[0] and prev_ball[1] >= current_ball[1]:
                new_trajectories.append(current_ball)
        elif prev_ball is None:
            new_trajectories.append(None)

    # Ensure the last ball is added
    new_trajectories.append(trajectories[-1])

    # Post-process the end of the second half of the shot
    while len(new_trajectories) > 3 and all(x is None for x in new_trajectories[-3:]):
        new_trajectories = new_trajectories[:-3]

    return new_first_half, new_trajectories


def extract_ball_trajectory(frames_data):
    basket = find_basket(frames_data)
    top_ball, top_frame = find_highest_ball_position(frames_data)
    ball_release_frame = find_ball_release(frames_data, top_frame)
    shot_flight_end_frame = find_shot_flight_end(frames_data, top_frame)
    if (ball_release_frame is None or ball_release_frame == top_frame) or (
        shot_flight_end_frame is None or shot_flight_end_frame == top_frame
    ):
        return None, None, None, None, None, None, None, None

    first_half_trajectories = []
    above_head_trajectories_half_trajectories = []

    # From ball release to top frame
    for frame in range(ball_release_frame, top_frame + 1):
        person_kepoints = get_person_keypoints_from_frame(frames_data[frame])
        person_bbox = get_person_bbox_from_frame(frames_data[frame])
        ball = get_top_ball_from_frame(frames_data[frame], basket)

        if ball is not None and is_ball_above_shoulders(ball, person_kepoints):
            first_half_trajectories.append(ball)
        else:
            if ball is None and len(first_half_trajectories) > 0:
                first_half_trajectories.append(None)

        if (
            ball is not None
            and is_ball_above_head_and_hands(ball, person_kepoints)
            and is_ball_to_left_of_person(ball, person_bbox, True)
        ):
            above_head_trajectories_half_trajectories.append(ball)
        else:
            if ball is None and len(above_head_trajectories_half_trajectories) > 0:
                above_head_trajectories_half_trajectories.append(None)

    # Ensure the highest ball is added
    first_half_trajectories.append(top_ball)
    above_head_trajectories_half_trajectories.append(top_ball)

    # Post-process the first half of the shot
    i = 0
    while i < len(first_half_trajectories) - 1:
        current_ball = first_half_trajectories[i]
        next_ball = first_half_trajectories[i + 1]

        if next_ball is not None and current_ball is not None:
            if next_ball[0] < current_ball[0] or next_ball[1] > current_ball[1]:
                if i == 0:
                    first_half_trajectories.pop(i)
                    continue
                else:
                    first_half_trajectories[i + 1] = None
        i += 1

    while len(first_half_trajectories) > 3 and all(x is None for x in first_half_trajectories[:3]):
        first_half_trajectories = first_half_trajectories[2:]

    # From top frame to shot flight end
    second_half_trajectories = []
    above_head_second_half_trajectories = []

    for frame in range(top_frame + 1, shot_flight_end_frame + 1):
        ball = get_top_ball_from_frame(frames_data[frame], basket)
        if ball is not None and is_ball_above_basket(ball, basket):
            second_half_trajectories.append(ball)
            above_head_second_half_trajectories.append(ball)
        else:
            if ball is None and len(second_half_trajectories) > 0:
                second_half_trajectories.append(None)
                above_head_second_half_trajectories.append(None)

    # Post-process the second half of the shot
    i = len(second_half_trajectories) - 1
    while i > 0:
        current_ball = second_half_trajectories[i]
        prev_ball = second_half_trajectories[i - 1]

        if prev_ball is not None and current_ball is not None:
            if prev_ball[0] < current_ball[0] or prev_ball[1] < current_ball[1]:
                if i == len(second_half_trajectories) - 1:
                    second_half_trajectories.pop(i)
                else:
                    second_half_trajectories[i - 1] = None
        i -= 1

    while len(second_half_trajectories) > 3 and all(x is None for x in second_half_trajectories[-3:]):
        second_half_trajectories = second_half_trajectories[:-3]

    # Combine first and second half to get full trajectories
    trajectories = first_half_trajectories + second_half_trajectories
    above_head_trajectories = above_head_trajectories_half_trajectories + above_head_second_half_trajectories

    return (
        trajectories,
        first_half_trajectories,
        above_head_trajectories,
        above_head_trajectories_half_trajectories,
        basket,
        ball_release_frame,
        shot_flight_end_frame,
        top_frame,
    )


def validate_trajectories(trajectories, top_frame):
    if trajectories is None:
        return False
    not_none_trajectories = [trajectory for trajectory in trajectories[:top_frame] if trajectory is not None]
    if len(not_none_trajectories) >= len(trajectories[:top_frame]) / 2 and len(not_none_trajectories) >= 8:
        # check if no more than 4 consecutive frames are None
        consecutive_none = 0
        for trajectory in trajectories[:top_frame]:
            if trajectory is None:
                consecutive_none += 1
                if consecutive_none > 3:
                    return False
            else:
                consecutive_none = 0
        return True


def visualize_trajectory(basket, ball_trajectory_frames, filename=None):
    ball_trajectory_frames = interpolate_positions(ball_trajectory_frames)

    # Create a blank 1920x1080 canvas
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)  # Reverse the y-axis to match video coordinates

    # Plot the basket position
    basket_x = (basket[0] + basket[2]) / 2
    basket_y = (basket[1] + basket[3]) / 2 + 100  # Shift down by 50
    ax.scatter(basket_x, basket_y, color="red", label="Basket", s=100)

    # Plot the ball trajectory
    ball_x = [ball[0] for ball in ball_trajectory_frames if ball is not None]
    ball_y = [ball[1] + 100 for ball in ball_trajectory_frames if ball is not None]  # Shift down by 50
    ax.plot(ball_x, ball_y, color="blue", label="Ball Trajectory", linewidth=2)  # Changed to line plot

    ax.legend()
    plt.title("Ball Trajectory Visualization")

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

    # Close the figure to free up memory
    plt.close(fig)


def interpolate_positions(positions, method="quadratic"):
    x1, y1, x2, y2 = zip(
        *[(p[0], p[1], p[2], p[3]) if p is not None else (None, None, None, None) for p in positions]
    )

    known_indices = [i for i, p in enumerate(positions) if p is not None]

    x1 = interp1d(
        known_indices,
        [x1[i] for i in known_indices],
        kind=method,
        fill_value="extrapolate",
    )(range(len(positions)))
    y1 = interp1d(
        known_indices,
        [y1[i] for i in known_indices],
        kind=method,
        fill_value="extrapolate",
    )(range(len(positions)))
    x2 = interp1d(
        known_indices,
        [x2[i] for i in known_indices],
        kind=method,
        fill_value="extrapolate",
    )(range(len(positions)))
    y2 = interp1d(
        known_indices,
        [y2[i] for i in known_indices],
        kind=method,
        fill_value="extrapolate",
    )(range(len(positions)))

    interpolated_positions = [(x1[i], y1[i], x2[i], y2[i]) for i in range(len(positions))]

    return interpolated_positions


def is_ball_above_head_and_hands(ball, person_keypoints):
    if len(person_keypoints) >= 11:
        head_y = person_keypoints[0][1]
        hands_y = (person_keypoints[9][1] + person_keypoints[10][1]) / 2
        if ball is not None:
            avg_ball_y = (ball[1] + ball[3]) / 2
            return avg_ball_y < head_y - 50 and avg_ball_y < hands_y - 50


def normalize_positions(ball_positions, basket):
    normalized_positions = []

    basket_x = (basket[0] + basket[2]) / 2
    basket_y = (basket[1] + basket[3]) / 2

    for ball_position in ball_positions:
        ball_x = (ball_position[0] + ball_position[2]) / 2
        ball_y = (ball_position[1] + ball_position[3]) / 2

        normalized_x = ball_x - basket_x
        normalized_y = ball_y - basket_y

        normalized_positions.append((normalized_x, normalized_y))

    return normalized_positions


def compute_angles(normalized_positions):
    angles = []

    for position in normalized_positions:
        angle = math.atan2(position[1], position[0])
        angles.append(angle)  # Convert to degrees

    return angles
