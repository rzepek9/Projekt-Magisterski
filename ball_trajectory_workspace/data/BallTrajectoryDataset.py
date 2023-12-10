import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class BallTrajectoryDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        first_half_only: bool = True,
        angles_only: bool = False,
        trajectories_only: bool = True,
        use_distance: bool = False,
        label_type: str = "made",
    ) -> None:
        valid_labels = ["made", "clean", "skill"]
        if label_type not in valid_labels:
            raise ValueError(f"label_type should be one of {valid_labels}. Got {label_type}.")

        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.features = []
        for key, value in self.data.items():
            prefix = "trajectories_half_" if first_half_only else ""

            x_values, y_values = zip(*value[f"normalized_above_head_{prefix}trajectories"])
            distances = [self.compute_distance(x, y) for x, y in zip(x_values, y_values)]
            x_values, y_values, distances = (
                self.normalize_data(x_values),
                self.normalize_data(y_values),
                self.normalize_data(distances),
            )

            if trajectories_only:
                feature_data = list(zip(x_values, y_values))
            elif angles_only:
                angles = value[f"{prefix}angles"]
                feature_data = [[angle] for angle in angles]
            else:
                angles = value[f"{prefix}angles"]
                feature_data = list(zip(x_values, y_values, angles))

            if use_distance:
                feature_data = [list(data) + [dist] for data, dist in zip(feature_data, distances)]

            self.features.append(torch.tensor(feature_data).float())

        self.labels = [torch.tensor(value[label_type]).float() for value in self.data.values()]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y, x.size(0)

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
    def collate_fn(batch: list) -> tuple:
        batch.sort(key=lambda x: x[0].size(0), reverse=True)
        features, labels, lengths = zip(*batch)
        features = pad_sequence(features, batch_first=True)
        labels = torch.stack(labels).unsqueeze(1) if labels[0] is not None else None
        lengths = torch.tensor(lengths)

        return features, labels, lengths
