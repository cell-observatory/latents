import unittest
import os
import tempfile
import shutil
import numpy as np
import tensorstore as ts
import torch
from pathlib import Path
from data.membrane_histone_dataset import MembraneHistoneDataset

class TestMembraneHistoneDataset(unittest.TestCase):
    def setUp(self):
        """Set up test data directory with mock zarr files."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()

        # Create mock data with different shapes
        self.store_shapes = [
            (10, 256, 256, 256, 2),  # Valid store
            (1, 256, 256, 256, 2),   # Too few time points
            (20, 63, 63, 63, 2),     # Too small spatial dimensions
            (15, 512, 512, 512, 2),  # Valid store
        ]

        # Create mock zarr files
        for i, shape in enumerate(self.store_shapes):
            # Create random data in uint16 range (0-65535)
            data = np.random.randint(0, 65536, size=shape, dtype=np.uint16)

            # Save as zarr
            store_path = os.path.join(self.test_dir, f'test_data_{i}.zarr')
            chunk_shape = [1, 64, 64, 64, 2]

            spec = {
                'driver': 'zarr3',
                'kvstore': {
                    'driver': 'file',
                    'path': store_path
                },
                'metadata': {
                    'data_type': 'uint16',
                    'shape': shape,
                    'chunk_grid': {
                        'name': 'regular',
                        'configuration': {'chunk_shape': chunk_shape}
                    },
                    'codecs': [{
                        "name": "sharding_indexed",
                        "configuration": {
                            "chunk_shape": chunk_shape,
                            "codecs": [
                                {"name": "bytes", "configuration": {"endian": "little"}},
                                {"name": "blosc", "configuration": {
                                    "cname": "zstd",
                                    "clevel": 1,
                                    "blocksize": 0,
                                    "shuffle": "shuffle"
                                }}
                            ],
                            "index_codecs": [
                                {"name": "bytes", "configuration": {"endian": "little"}},
                                {"name": "crc32c"}
                            ],
                            "index_location": "end"
                        }
                    }],
                    'fill_value': 0
                },
                'create': True,
                'delete_existing': True
            }
            store = ts.open(spec).result()
            store.write(data).result()

    def tearDown(self):
        """Clean up test data directory."""
        shutil.rmtree(self.test_dir)

    def test_init_valid_stores(self):
        """Test initialization with valid stores."""
        dataset = MembraneHistoneDataset(
            data_path=self.test_dir,
            time_window=3,
            crop_size=(64, 64, 64)
        )

        # Should only load stores with sufficient dimensions
        self.assertEqual(len(dataset.stores), 2)

    def test_init_no_files(self):
        """Test initialization with no zarr files."""
        empty_dir = tempfile.mkdtemp()
        try:
            with self.assertRaises(ValueError):
                MembraneHistoneDataset(data_path=empty_dir)
        finally:
            shutil.rmtree(empty_dir)

    def test_getitem(self):
        """Test __getitem__ method."""
        dataset = MembraneHistoneDataset(
            data_path=self.test_dir,
            time_window=3,
            crop_size=(64, 64, 64)
        )

        # Test valid index
        data = dataset[0]
        self.assertIsInstance(data, torch.Tensor)
        self.assertEqual(data.dtype, torch.float32)  # Should be converted to float32
        self.assertEqual(data.shape[0], 6)  # 3 time points * 2 channels
        self.assertEqual(data.shape[1:], (64, 64, 64))

    def test_len(self):
        """Test __len__ method."""
        dataset = MembraneHistoneDataset(
            data_path=self.test_dir,
            time_window=3,
            crop_size=(64, 64, 64)
        )
        self.assertEqual(len(dataset), 2)  # Number of valid stores

    def test_transform(self):
        """Test dataset with transform."""
        def test_transform(x):
            return x * 2

        dataset = MembraneHistoneDataset(
            data_path=self.test_dir,
            time_window=3,
            crop_size=(64, 64, 64),
            transform=test_transform
        )

        data = dataset[0]
        # Check that transform was applied
        self.assertTrue(torch.all(data % 2 == 0))

    def test_different_crop_sizes(self):
        """Test dataset with different crop sizes."""
        crop_sizes = [(32, 32, 32), (64, 64, 64), (128, 128, 128)]

        for crop_size in crop_sizes:
            dataset = MembraneHistoneDataset(
                data_path=self.test_dir,
                time_window=3,
                crop_size=crop_size
            )

            data = dataset[0]
            self.assertEqual(data.shape[1:], crop_size)

    def test_different_time_windows(self):
        """Test dataset with different time windows."""
        time_windows = [1, 3, 5]

        for time_window in time_windows:
            dataset = MembraneHistoneDataset(
                data_path=self.test_dir,
                time_window=time_window,
                crop_size=(64, 64, 64)
            )

            data = dataset[0]
            self.assertEqual(data.shape[0], time_window * 2)  # time_window * channels

if __name__ == '__main__':
    unittest.main()