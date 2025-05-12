import unittest
import torch
from models.autoencoder import (
    ResnetBlock, ResnetBlockConfig,
    Downsample, DownsampleConfig
)

class TestAutoEncoder3D(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(123)

    def test_ResnetBlock3D(self):
        config = ResnetBlockConfig(dim=3, norm_num_groups=1)
        net = ResnetBlock(3, 4, config).to(self.device)
        x = torch.randn((1, 3, 32, 32, 32)).to(self.device)
        y = net(x)
        self.assertEqual(y.shape, (1, 4, 32, 32, 32))

    def test_Downsample3D(self):
        config = DownsampleConfig(dim=3)
        net = Downsample(3, 4, config).to(self.device)
        x = torch.randn((1, 3, 32, 32, 32)).to(self.device)
        y = net(x)
        self.assertEqual(y.shape, (1, 4, 16, 16, 16))

    # TODO: Perform other 3D tests


if __name__ == '__main__':
    unittest.main()
