import os
import torch
from typing import Optional, Callable, Tuple, Union, List
import numpy as np
from pathlib import Path
import logging
import tensorstore as ts
from torchvision import transforms

logger = logging.getLogger(__name__)

class MembraneHistoneDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path: str,
                 transform: Optional[Callable] = None,
                 time_window: int = 1,
                 crop_size: Tuple[int, int, int] = (128, 128, 128)):
        """
        Args:
            data_path: Path to directory containing zarr files
            transform: Optional transform to apply after random cropping. If None, no additional transform is applied.
            time_window: Number of consecutive time points to sample in each window.
            crop_size: Size of the 3D crop in (depth, height, width) format
        """
        self.data_path = Path(data_path)
        self.time_window = time_window
        self.crop_size = crop_size
        self.transform = transform

        # Recursively find all .zarr files in subdirectories
        self.data_files = sorted(list(self.data_path.rglob('*.zarr')))
        if not self.data_files:
            raise ValueError(f"No .zarr files found in {data_path} or its subdirectories")

        # Open and filter stores based on time points
        logger.info(f"Opening {len(self.data_files)} zarr files...")
        self.stores = []
        self.store_shapes = []
        self.store_paths = []

        for file_path in self.data_files:
            try:
                spec = {
                    'driver': 'zarr3',
                    'kvstore': {
                        'driver': 'file',
                        'path': str(file_path)
                    }
                }
                store = ts.open(spec).result()
                shape = store.shape

                # Validate store shape
                if len(shape) != 5:  # [T,D,H,W,C]
                    logger.warning(f"Skipping store {file_path} with invalid shape {shape}. Expected 5 dimensions [T,D,H,W,C]")
                    continue

                # Validate dimensions against crop size
                if any(c > s for c, s in zip(self.crop_size, shape[1:4])):
                    logger.warning(f"Skipping store {file_path} with insufficient spatial dimensions. "
                                 f"Shape: {shape}, Required crop: {self.crop_size}")
                    continue

                max_time = shape[0]
                if max_time >= self.time_window:
                    self.stores.append(store)
                    self.store_shapes.append(shape)
                    self.store_paths.append(file_path)
                    logger.info(f"Added store {file_path} with shape {shape}")
                else:
                    logger.warning(f"Skipping store {file_path} with insufficient time points "
                                 f"(has {max_time}, needs at least {self.time_window})")
            except Exception as e:
                logger.error(f"Error opening store {file_path}: {str(e)}")
                continue

        if not self.stores:
            raise ValueError(f"No valid stores found with sufficient dimensions for "
                           f"time_window={self.time_window} and crop_size={self.crop_size}")

        logger.info(f"Dataset contains {len(self.stores)} stores")
        logger.info(f"Store shapes: {self.store_shapes}")

    def _random_4d_crop(self, data: ts.TensorStore) -> torch.Tensor:
        """Randomly crop a 3D region and sample a window of time points.
        Returns tensor of shape [C*T,D,H,W] where T is the time window size.

        Args:
            data: Input tensorstore object of shape [T,D,H,W,C]
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

        # Reshape to combine time and channel dimensions
        # From [T,D,H,W,C] to [T*C,D,H,W]
        data = data.reshape(-1, *data.shape[1:4])  # [T*C,D,H,W]

        return data

    def __len__(self):
        return len(self.stores)

    def __getitem__(self, idx):
        if idx >= len(self.stores):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self.stores)}")

        # Get the store and apply random 4D crop
        data = self._random_4d_crop(self.stores[idx])

        # Apply additional transform if specified
        if self.transform:
            data = self.transform(data)

        return torch.from_numpy(data).float()
