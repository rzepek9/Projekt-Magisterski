import numpy as np
import torch
import wandb
import yaml
from data import FixedBallTrajectoryDataset
from models import Conv1DNet
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

DATA_PATH = "C:/Users/gustaw/studia/magisterka_filmy/ball_trajectory/extracted_trajectories.json"
SWEEP_PATH = "C:/Users/gustaw/studia/Projekt-Magisterski/ball_trajectory_workspace/configs/"


def get_sweep_for_label(label_type, kernel_size):
    sweep_filename = f"sweep_{label_type}_{kernel_size}.yaml"
    sweep_path = SWEEP_PATH + sweep_filename

    with open(sweep_path, 'r') as file:
        sweep_config = yaml.safe_load(file)

    return sweep_config


def train():
    wandb.init(project="BALL-TRAJECTORY_CONV1D")
    if wandb.config.angles_only and wandb.config.trajectories_only:
        print("Both angles_only and trajectories_only are set to True. Skipping this configuration.")
        return
    if wandb.config.scheduler == "CyclicLR" and wandb.config.optimizer != "SGD":
        print("CyclicLR scheduler requires SGD optimizer. Skipping this configuration.")
        return
    if not (len(wandb.config.conv_neurons) == len(wandb.config.kernel_sizes)):
        print("The length of conv_neurons and kernel_sizes list must be equal to n_conv_layers. Skipping this configuration.")
        return

    train_loader, val_loader = get_train_val_loaders(
        DATA_PATH,
        angles_only=wandb.config.angles_only,
        first_half_only=wandb.config.first_half_only,
        trajectories_only=wandb.config.trajectories_only,
        label_type=wandb.config.label_type,
        use_distance=wandb.config.use_distance,
        use_padding=wandb.config.use_padding,
        batch_size=wandb.config.batch_size,
        test_size=wandb.config.test_size,
    )

    # Determine input feature size
    if wandb.config.angles_only:
        input_size = 1  # Assuming 1 feature for angles
    elif wandb.config.trajectories_only:
        input_size = 2  # Assuming 2 features for trajectories (x and y)
    else:
        input_size = 3  # Assuming 3 features if both are used (angle, x, y)
    if wandb.config.use_distance:
        input_size += 1

    if wandb.config.use_padding:
        input_shape = (input_size, 14)
    elif wandb.config.first_half_only:
        input_shape = (input_size, 6)
    else:
        input_shape = (input_size, 7)

    # Determine output size based on label_type
    if wandb.config.label_type == "made":
        num_classes = 1
    elif wandb.config.label_type == "clean":
        num_classes = 1
    elif wandb.config.label_type == "skill":
        num_classes = 3  # Assuming 3 classes for skill
    else:
        raise ValueError(f"Unsupported label type: {wandb.config.label_type}")

    conv_layers = len(wandb.config.conv_neurons)
    model = Conv1DNet(
        input_shape=input_shape,
        n_conv_layers=conv_layers,
        hidden_size=wandb.config.hidden_size,
        num_classes=num_classes,
        dropout=wandb.config.dropout,
        kernel_sizes=wandb.config.kernel_sizes,
        skip_connection=wandb.config.skip_connection,
        conv_neurons=wandb.config.conv_neurons,
    )

    if wandb.config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.learning_rate)
    elif wandb.config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    elif wandb.config.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {wandb.config.optimizer}")

    train_model(
        model,
        train_loader,
        val_loader,
        wandb.config.epochs,
        wandb.config.patience,
        optimizer,
        scheduler_type=wandb.config.scheduler,
        label_type=wandb.config.label_type,
    )


def train_model(
    model,
    train_loader,
    val_loader,
    epochs,
    patience=10,
    optimizer=None,
    scheduler_type="ReduceLROnPlateau",
    label_type="skill",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optimizer

    if scheduler_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.00001)
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    elif scheduler_type == "CyclicLR":
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.000001, max_lr=0.01)
    else:
        raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")

    best_val_loss = float("inf")
    best_epoch = 0
    best_model = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, label_type, scheduler)
        val_loss, val_metrics = evaluate_model(model, val_loader, device, label_type)
        if scheduler_type == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        wandb.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": val_metrics["f1"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "epoch": epoch,
            }
        )

        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Metrics: {val_metrics}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model = model.state_dict()

        if epoch - best_epoch > patience:
            print("Early stopping")
            break

    torch.save(best_model, "best_model.pt")


def train_one_epoch(model, train_loader, optimizer, device, label_type="skill", scheduler=None):
    model.train()
    train_loss = 0.0

    if label_type == "skill":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if label_type == "skill":
            labels = labels.squeeze(1).long()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
            scheduler.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    return train_loss


def evaluate_model(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, label_type="skill"
) -> tuple:
    model.eval()
    loss = 0.0
    predictions = []
    true_labels = []

    if label_type == "skill":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            if label_type == "skill":
                labels = labels.squeeze(1).long()
            loss += criterion(outputs, labels).item()
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    loss /= len(loader)
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    if label_type == "skill":
        predictions_binary = np.argmax(predictions, axis=1)
    else:
        predictions_binary = (predictions >= 0.5).astype(int)
    accuracy = accuracy_score(true_labels, predictions_binary)
    precision, recall, f1, _ = (
        precision_recall_fscore_support(true_labels, predictions_binary, average="binary")
        if label_type != "skill"
        else precision_recall_fscore_support(true_labels, predictions_binary, average="macro")
    )
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }
    return loss, metrics


def get_train_val_loaders(
    data_path: str,
    angles_only: bool = False,
    first_half_only: bool = True,
    trajectories_only: bool = True,
    use_distance: bool = False,
    use_padding: bool = False,
    label_type: str = "made",
    batch_size: int = 16,
    test_size: float = 0.25,
) -> tuple:
    dataset = FixedBallTrajectoryDataset(
        data_path,
        first_half_only=first_half_only,
        angles_only=angles_only,
        trajectories_only=trajectories_only,
        use_distance=use_distance,
        use_padding=use_padding,
        label_type=label_type,
    )

    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=test_size, random_state=42)

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    label_types = ["made", "skill", "clean"]
    kernel_sizes = [2, 3]

    for label_type in label_types:
        for kernel_size in kernel_sizes:
            sweep_config = get_sweep_for_label(label_type, kernel_size)
            sweep_id = wandb.sweep(sweep_config)
            wandb.agent(sweep_id, function=train, count=50)
