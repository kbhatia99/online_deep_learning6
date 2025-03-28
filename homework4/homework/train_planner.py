import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from .models import load_model, save_model
from homework.datasets.road_dataset import load_data
from torch.nn import functional as F

class TrackWaypointDataset(Dataset):
    def __init__(self, data_list, n_track=10, n_waypoints=3):
        super().__init__()
        self.data = data_list
        self.n_track = n_track
        self.n_waypoints = n_waypoints

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        return (
            torch.tensor(entry['track_left'], dtype=torch.float32),
            torch.tensor(entry['track_right'], dtype=torch.float32),
            torch.tensor(entry['waypoints'], dtype=torch.float32),
            torch.tensor(entry['waypoints_mask'], dtype=torch.bool),
        )

def masked_mse_loss(pred, target, mask):
    """
    Compute mean squared error (L2) only on valid waypoints (mask == True).
    """
    mask = mask.unsqueeze(-1)
    diff = (pred - target) * mask
    loss = (diff ** 2).sum() / (mask.sum() + 1e-6)
    return loss

def masked_l1_loss(pred, target, mask):
    """
    Compute mean absolute error (L1) only on valid waypoints (mask == True).
    """
    mask = mask.unsqueeze(-1)
    diff = (pred - target) * mask
    loss = diff.abs().sum() / (mask.sum() + 1e-6)
    return loss

def train(
    model_name="mlp_planner",
    transform_pipeline="state_only",
    dataset_path="/content/online_deep_learning6/homework4/drive_data/train",
    num_epochs=30,
    lr=1e-3,
    num_workers=4,
    batch_size=64,
    weight_factor=1e-4,  # Regularization weight
):
    loader = load_data(
        dataset_path=dataset_path,
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
    )

    model = load_model(model_name)
    model = model.cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in loader:
            data = {k: v.cuda() for k, v in batch.items()}

            if "image" in data:
                pred = model(data["image"])
            else:
                pred = model(track_left=data["track_left"], track_right=data["track_right"])

            # Use L1 loss + L2 regularization
            loss = masked_l1_loss(pred, data["waypoints"], data["waypoints_mask"])
            l2_reg = sum((param ** 2).sum() for param in model.parameters())
            loss = loss + weight_factor * l2_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}] Loss: {total_loss / len(loader):.4f}")

    save_path = save_model(model)
    print(f"✅ Model saved to {save_path}")


if __name__ == "__main__":
    from random import random
    dummy_data = [
        {
            "track_left": [[random(), random()] for _ in range(10)],
            "track_right": [[random(), random()] for _ in range(10)],
            "waypoints": [[random(), random()] for _ in range(3)],
            "waypoints_mask": [True, True, True]
        }
        for _ in range(1000)
    ]
    train(model_name="transformer_planner", data_list=dummy_data)

print("Time to train")
