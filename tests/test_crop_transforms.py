import unittest
import numpy as np
import tensorstore as ts
import tempfile
from pathlib import Path
from data.crop_transforms import Random4DCrop

class TestRandom4DCrop(unittest.TestCase):
    def setUp(self):
        """Set up test data before each test."""
        # Create test data
        self.time_points = 5
        self.depth = 100
        self.height = 100
        self.width = 100
        self.channels = 2

        data = np.random.randint(0, 65536, size=(self.time_points, self.depth, self.height, self.width, self.channels), dtype=np.uint16)

        # Create temporary directory using tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix="test_crop_transforms_"))
        store_path = str(temp_dir / "test_data.zarr")

        # Define chunk shape
        chunk_shape = (1, self.depth//2, self.height//2, self.width//2, self.channels)

        # Create tensorstore spec
        spec = {
            'driver': 'zarr3',
            'kvstore': {
                'driver': 'file',
                'path': store_path
            },
            'metadata': {
                'data_type': 'uint16',
                'shape': data.shape,
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

        # Create and write to tensorstore
        store = ts.open(spec).result()
        store.write(data).result()

        self.data = store

        # Initialize transform
        self.transform = Random4DCrop(
            time_window=2,
            crop_size=(32, 32, 32)
        )

    def test_output_shape(self):
        """Test that output has correct shape."""
        output = self.transform(self.data)
        expected_shape = (self.transform.time_window * self.channels,
                         self.transform.crop_size[0],
                         self.transform.crop_size[1],
                         self.transform.crop_size[2])
        self.assertEqual(output.shape, expected_shape)

    def test_normalization_range(self):
        """Test that output values are in [0,1] range."""
        output = self.transform(self.data)
        self.assertGreaterEqual(output.min(), 0)
        self.assertLessEqual(output.max(), 1)

        # Check that each channel is normalized independently
        for c in range(self.channels):
            channel_data = output[c::self.channels]  # Get all time points for this channel
            self.assertAlmostEqual(channel_data.min(), 0.)
            self.assertAlmostEqual(channel_data.max(), 1.)

    def test_invalid_time_window(self):
        """Test that invalid time window raises error."""
        with self.assertRaises(ValueError):
            transform = Random4DCrop(time_window=self.time_points + 1)
            transform(self.data)

    def test_invalid_crop_size(self):
        """Test that invalid crop size raises error."""
        with self.assertRaises(ValueError):
            transform = Random4DCrop(crop_size=(self.depth + 1, 32, 32))
            transform(self.data)


if __name__ == '__main__':
    unittest.main()