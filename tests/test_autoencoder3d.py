import unittest
import torch
from models.autoencoder import (
    ResnetBlock, ResnetBlockConfig,
    Downsample, DownsampleConfig,
    Upsample, UpsampleConfig,
    DownEncoderBlock, DownEncoderBlockConfig,
    UpDecoderBlock, UpDecoderBlockConfig,
    UNetMidBlock, UNetMidBlockConfig,
    Encoder, EncoderConfig
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

    def test_Upsample3D(self):
        config = UpsampleConfig(dim=3)
        net = Upsample(3, 4, config).to(self.device)
        x = torch.randn((1, 3, 16, 16, 16)).to(self.device)
        y = net(x)
        self.assertEqual(y.shape, (1, 4, 32, 32, 32))

    def test_DownEncoderBlock3D(self):
        resnet_block_config = ResnetBlockConfig(dim=3, norm_num_groups=1)
        downsample_config = DownsampleConfig(dim=3)
        config = DownEncoderBlockConfig(resnet_block_config=resnet_block_config, downsample_config=downsample_config)
        
        # Downsample once
        n_downsamplers = 1
        net = DownEncoderBlock([3, 4, 4, 4], n_downsamplers, config).to(self.device)
        x = torch.randn((1, 3, 32, 32, 32)).to(self.device)
        y = net(x)
        self.assertEqual(y.shape, (1, 4, 16, 16, 16))

        # Downsample twice
        n_downsamplers = 2
        net = DownEncoderBlock([3, 4, 4, 4], n_downsamplers, config).to(self.device)
        x = torch.randn((1, 3, 32, 32, 32)).to(self.device)
        y = net(x)
        self.assertEqual(y.shape, (1, 4, 8, 8, 8))

    def test_UpDecoderBlock3D(self):
        resnet_block_config = ResnetBlockConfig(dim=3, norm_num_groups=1)
        upsample_config = UpsampleConfig(dim=3)
        config = UpDecoderBlockConfig(resnet_block_config=resnet_block_config, upsample_config=upsample_config)

        # Upsample once
        n_upsamplers = 1    
        net = UpDecoderBlock([3, 4, 4, 4], n_upsamplers, config).to(self.device)
        x = torch.randn((1, 3, 8, 8, 8)).to(self.device)
        y = net(x)
        self.assertEqual(y.shape, (1, 4, 16, 16, 16))

        # Upsample twice
        n_upsamplers = 2
        net = UpDecoderBlock([3, 4, 4, 4], n_upsamplers, config).to(self.device)
        x = torch.randn((1, 3, 8, 8, 8)).to(self.device)
        y = net(x)
        self.assertEqual(y.shape, (1, 4, 32, 32, 32))

    def test_UNetMidBlock3D(self):
        resnet_block_config = ResnetBlockConfig(dim=3, norm_num_groups=1)
        config = UNetMidBlockConfig(resnet_block_config=resnet_block_config)
        net = UNetMidBlock(1, 512, config).to(self.device)
        x = torch.randn((1, 512, 8, 8, 8)).to(self.device)
        y = net(x)
        self.assertEqual(y.shape, (1, 512, 8, 8, 8))

    def test_encoder(self):
        resnet_block_config = ResnetBlockConfig(dim=3, norm_num_groups=1)
        downsample_config = DownsampleConfig(dim=3)
        unet_mid_block_config = UNetMidBlockConfig(resnet_block_config=resnet_block_config)
        down_encoder_block_config = DownEncoderBlockConfig(resnet_block_config=resnet_block_config, downsample_config=downsample_config)
        encoder_config = EncoderConfig(down_encoder_block_config=down_encoder_block_config,
                                       unet_mid_block_config=unet_mid_block_config,
                                       conv_in_dim=3,
                                       conv_out_dim=3,
                                       )
        encoder = Encoder(config=encoder_config).to(self.device)
        x = torch.randn((1, 3, 64, 64, 64)).to(self.device)
        y = encoder(x)
        self.assertEqual(y.shape, (1, 8, 8, 8, 8))

    # TODO: test decoder

if __name__ == '__main__':
    unittest.main()
