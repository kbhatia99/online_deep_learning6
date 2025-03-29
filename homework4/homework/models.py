from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden_dim: int = 128,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
            hidden_dim (int): size of hidden layer in MLP
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_dim = 4 * n_track  # track_left (b, 10, 2) + track_right (b, 10, 2) → (b, 40)
        output_dim = n_waypoints * 2  # 3 waypoints × (x, y)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        b = track_left.size(0)

        # Flatten and concatenate left and right tracks: (b, 10, 2) → (b, 20)
        x = torch.cat([track_left, track_right], dim=1)  # (b, 20, 2)
        x = x.view(b, -1)  # (b, 40)

        # MLP to predict waypoints
        out = self.mlp(x)  # (b, 6) if 3 waypoints

        # Reshape output to (b, 3, 2)
        return out.view(b, self.n_waypoints, 2)


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Positional encodings can be learned or sinusoidal
        self.input_proj = nn.Linear(2, d_model)  # from (x, y) → d_model
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Transformer Decoder-only architecture (Encoder is track data)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final projection to (x, y) coordinates
        self.output_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,   # shape (b, n_track, 2)
        track_right: torch.Tensor,  # shape (b, n_track, 2)
        **kwargs,
    ) -> torch.Tensor:
        b = track_left.size(0)

        # 1. Concatenate track boundaries → (b, 2*n_track, 2)
        track = torch.cat([track_left, track_right], dim=1)

        # 2. Project (x, y) → d_model
        memory = self.input_proj(track)  # shape (b, 2*n_track, d_model)

        # 3. Encode the track
        memory = self.encoder(memory)  # shape (b, 2*n_track, d_model)

        # 4. Generate query embeddings for the decoder
        query_positions = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)  # shape (b, n_waypoints, d_model)

        # 5. Decode into future waypoint representations
        decoded = self.decoder(query_positions, memory)  # shape (b, n_waypoints, d_model)

        # 6. Project to 2D waypoints
        waypoints = self.output_proj(decoded)  # shape (b, n_waypoints, 2)

        return waypoints


# Example normalization values (ImageNet stats; you can replace them with your own dataset's stats)
INPUT_MEAN = [0.485, 0.456, 0.406]
INPUT_STD = [0.229, 0.224, 0.225]

class CNNPlanner(nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()
        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Simple CNN backbone with a large kernel in the first layer
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=16, stride=2, padding=7),  # -> (b, 32, h/2, w/2)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # -> (b, 64, h/4, w/4)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (b, 128, h/8, w/8)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # NEW layer → (b, 128, h/8, w/8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> (b, 64, 1, 1)
        )

        # Fully connected head to predict waypoints
        self.fc = nn.Sequential(
            nn.Flatten(),  # -> (b, 128)
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(64, n_waypoints * 2)  # output is (x, y) pairs
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w), values in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Normalize image
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Forward through CNN
        x = self.backbone(x)
        x = self.fc(x)
        return x.view(x.size(0), self.n_waypoints, 2)



MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
