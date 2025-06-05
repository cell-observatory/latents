import torch
from typing import Optional, Callable, Tuple
from pathlib import Path
import logging
import tensorstore as ts

logger = logging.getLogger(__name__)

class MembraneHistoneDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path: str,
                 transform: Optional[Callable] = None,
                 ):
        """
        Args:
            data_path: Path to directory containing zarr files
            transform: Optional transform to apply. Default is None.
        """
        self.data_path = Path(data_path)
        self.transform = transform
        
        # Find all .zarr files in subdirectories
        self.data_files = sorted(list(self.data_path.glob('*/*/*.zarr')))
        if not self.data_files:
            raise ValueError(f"No .zarr files found in {data_path} or its subdirectories")

        # Open and filter stores based on time points
        logger.info(f"Opening {len(self.data_files)} zarr files...")
        self.stores = []

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
                else:
                    self.stores.append(store)
            except Exception as e:
                logger.error(f"Error opening store {file_path}: {str(e)}")
                continue
        
        if not self.stores:
            raise ValueError(f"No valid stores found with the correct dimensions")

    def __len__(self):
        return len(self.stores)

    def __getitem__(self, idx):
        if idx >= len(self.stores):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self.stores)}")

        data = self.stores[idx]
        if self.transform:
            data = self.transform(data)

        return data

if __name__ == "__main__":
    data_path = '/CellObservatoryData/20250324_mem_histone'
    dataset = MembraneHistoneDataset(data_path=data_path)
    print(f"MembraneHistoneDataset constructed with {len(dataset)} many stores")