import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional, Dict
import tensorstore as ts

class Random4DCrop(nn.Module):
    def __init__(self,
                 time_window: int = 1,
                 crop_size: Tuple[int, int, int] = (128, 128, 128)):
        """
        Custom transform that performs random 4D cropping (3D spatial + time) with min-max normalization.

        Args:
            time_window: Number of consecutive time points to sample in each window
            crop_size: Size of the 3D crop in (depth, height, width) format
        """
        super().__init__()
        self.time_window = time_window
        self.crop_size = crop_size

    def _normalize_crop(self, crop: torch.Tensor) -> torch.Tensor:
        """
        Normalize a crop to [0,1] range.
        Normalizes across all time points for each channel.

        Args:
            crop: Input tensor of shape [T,D,H,W]

        Returns:
            Normalized tensor of same shape in [0,1] range
        """
        min_val = crop.min()
        max_val = crop.max()
        # Add small epsilon to avoid division by zero
        return (crop - min_val) / (max_val - min_val + 1e-6)

    def forward(self, data: ts.TensorStore) -> torch.Tensor:
        """
        Apply random 4D crop and normalize the data.

        Args:
            data: Input tensorstore object of shape [T,D,H,W,C]

        Returns:
            Normalized tensor of shape [C*T,D,H,W] with values in [0,1] range
        """
        # Get random time indices to sample
        max_time = data.shape[0]
        max_start = max_time - self.time_window
        if max_start < 0:
            raise ValueError(f"Time window of size {self.time_window} "
                           f"is too large for time dimension {max_time}")
        start_time = np.random.randint(0, max_start + 1)

        # Calculate maximum possible starting indices for spatial dimensions
        max_d = data.shape[1] - self.crop_size[0]
        max_h = data.shape[2] - self.crop_size[1]
        max_w = data.shape[3] - self.crop_size[2]

        # Generate random starting indices
        start_d = np.random.randint(0, max_d + 1)
        start_h = np.random.randint(0, max_h + 1)
        start_w = np.random.randint(0, max_w + 1)

        # Create slice objects for all dimensions
        full_slice = (
            slice(start_time, start_time + self.time_window),
            slice(start_d, start_d + self.crop_size[0]),
            slice(start_h, start_h + self.crop_size[1]),
            slice(start_w, start_w + self.crop_size[2]),
            slice(None)  # Take all channels
        )

        # Read data
        data = data[full_slice].read().result()

        # Convert to torch tensor
        data = torch.from_numpy(data).float()  # [T,D,H,W,C]

        # Channels first for easier processing
        data = data.permute(4, 0, 1, 2, 3)  # [C,T,D,H,W]

        # Process each channel separately, normalizing across all time points
        normalized_data = []
        for c in range(data.shape[0]):  # Iterate over channels
            channel_data = data[c]  # [T,D,H,W]
            normalized_channel = self._normalize_crop(channel_data)  # [T,D,H,W]
            normalized_data.append(normalized_channel)

        # Stack all channels and reshape to [T*C,D,H,W]
        normalized_data = torch.stack(normalized_data)  # [C,T,D,H,W]
        return normalized_data.reshape(-1, *normalized_data.shape[2:])  # [C*T,D,H,W]