import json
import torch
import numpy as np
from torch.utils.data import Dataset

MIN_TRAJECTORY_LENGTH = 7
MIN_FIRST_HALF_TRAJECTORY_LENGTH = 6
DATA_PADDING_VALUE = -0.1
ANGLE_PADDING_VALUE = -3.2
TRAJECTORIES_WITH_PADDING_LENGTH = 14


class ExperimentalBallTrajectoryDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        first_half_only: bool = True,
        angles_only: bool = False,
        trajectories_only: bool = True,
        use_distance: bool = False,
        use_padding: bool = False,
        label_type: str = "made",
    ) -> None:
        if use_padding:
            self.fixed_length = TRAJECTORIES_WITH_PADDING_LENGTH
        elif first_half_only:
            self.fixed_length = MIN_FIRST_HALF_TRAJECTORY_LENGTH
        else:
            self.fixed_length = MIN_TRAJECTORY_LENGTH

        valid_labels = ["made", "clean", "skill"]
        if label_type not in valid_labels:
            raise ValueError(f"label_type should be one of {valid_labels}. Got {label_type}.")

        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.features = []
        for key, value in self.data.items():
            prefix = "half_" if first_half_only else ""

            x_values, y_values = zip(*value[f"normalized_above_head_{prefix}trajectories"])
            distances = [self.compute_distance(x, y) for x, y in zip(x_values, y_values)]
            x_values, y_values, distances = (
                self.normalize_data(x_values),
                self.normalize_data(y_values),
                self.normalize_data(distances),
            )

            feature_data = []
            if trajectories_only:
                feature_data.append(list(x_values))
                feature_data.append(list(y_values))
            elif angles_only:
                angles = value[f"above_head_{prefix}angles"]
                feature_data.append(angles)
            else:
                feature_data.append(list(x_values))
                feature_data.append(list(y_values))
                feature_data.append(value[f"above_head_{prefix}angles"])

            if use_distance:
                feature_data.append(distances)

            feature_tensor = torch.tensor(feature_data).float()
            self.features.append(feature_tensor)

        self.labels = [torch.tensor(value[label_type]).float() for value in self.data.values()]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        x = self.uniform_sample(x, self.fixed_length)
        y = self.labels[idx].unsqueeze(-1)
        return x, y

    @staticmethod
    def normalize_data(data):
        """Apply Min-Max scaling to a list of values."""
        min_val, max_val = min(data), max(data)
        return [(val - min_val) / (max_val - min_val) for val in data]

    @staticmethod
    def compute_distance(x, y):
        """Compute the Euclidean distance from the basket (0,0)."""
        return (x**2 + y**2) ** 0.5

    @staticmethod
    def pad_tensor(tensor, target_length, padding_value):
        """Pad tensor to target length."""
        padding_size = target_length - tensor.size(1)
        padding = torch.full((tensor.size(0), padding_size), padding_value)
        return torch.cat([tensor, padding], dim=1)

    @staticmethod
    def uniform_sample(tensor, fixed_length):
        current_length = tensor.size(1)

        # If tensor is shorter than required, pad it
        if current_length < fixed_length:
            # Check if tensor contains angles and use appropriate padding value
            padding_value = ANGLE_PADDING_VALUE if tensor.min() < 0 else DATA_PADDING_VALUE
            return FixedBallTrajectoryDataset.pad_tensor(tensor, fixed_length, padding_value)

        # If tensor is longer than required, sample it uniformly
        indices = torch.linspace(0, current_length - 1, fixed_length).long()
        return tensor[:, indices]
