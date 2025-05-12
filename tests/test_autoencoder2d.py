import unittest
import torch
from diffusers import AutoencoderKL
from models.autoencoder import (
    ResnetBlock, Downsample, Upsample, DownEncoderBlock, UpDecoderBlock, UNetMidBlock,
    Encoder, Decoder, AutoEncoder
)

class TestAutoEncoder2DAgainstDiffusers(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"  # can also be a local file
        self.model = AutoencoderKL.from_single_file(url).to(self.device)
        torch.manual_seed(123)

    def check_load_state_dict_output(self, result):
        self.assertEqual(len(result[0]), 0, "Missing keys detected.")
        self.assertEqual(len(result[1]), 0, "Unexpected keys detected.")

    def check_output(self, net, test, x, temb = False, msg=""):
        if temb:
            y1 = test(x, None)
        else:
            y1 = test(x)
        y2 = net(x)
        self.assertTrue(torch.equal(y1, y2), msg + "Output tensors must match for the same input.")

    def test_ResnetBlock2D(self):
        # Without shortcut
        test = self.model.encoder.down_blocks[0].resnets[0]
        net = ResnetBlock(128,128).to(self.device)
        result = net.load_state_dict(test.state_dict())
        self.check_load_state_dict_output(result)
        x = torch.rand(1, 128, 8, 8).to(self.device)
        self.check_output(net, test, x, temb = True)

        # With shortcut
        test = self.model.encoder.down_blocks[1].resnets[0]
        net = ResnetBlock(128, 256).to(self.device)
        result = net.load_state_dict(test.state_dict())
        self.check_load_state_dict_output(result)
        x = torch.rand(1, 128, 8, 8).to(self.device)
        self.check_output(net, test, x, temb = True)

    def test_Downsample2D(self):
        test = self.model.encoder.down_blocks[0].downsamplers[0]
        net = Downsample(128,128).to(self.device)
        result = net.load_state_dict(test.state_dict())
        self.check_load_state_dict_output(result)
        x = torch.rand(1, 128, 8, 8).to(self.device)
        self.check_output(net, test, x)

    def test_Upsample2D(self):
        test = self.model.decoder.up_blocks[0].upsamplers[0]
        net = Upsample(512,512).to(self.device)
        result = net.load_state_dict(test.state_dict())
        self.check_load_state_dict_output(result)
        x = torch.rand(1, 512, 8, 8).to(self.device)
        self.check_output(net, test, x)

    def test_DownEncoderBlock2D(self):
        test = self.model.encoder.down_blocks[1]
        net = DownEncoderBlock([128, 256, 256]).to(self.device)
        result = net.load_state_dict(test.state_dict())
        self.check_load_state_dict_output(result)
        x = torch.rand(1, 128, 16, 16).to(self.device)
        self.check_output(net, test, x)

    def test_UpDecoderBlock2D(self):
        test = self.model.decoder.up_blocks[2]
        net = UpDecoderBlock([512, 256, 256, 256]).to(self.device)
        result = net.load_state_dict(test.state_dict())
        self.check_load_state_dict_output(result)
        x = torch.rand(1, 512, 16, 16).to(self.device)
        self.check_output(net, test, x)

    def test_UNetMidBlock2D(self):
        # Encoder
        test = self.model.encoder.mid_block
        net = UNetMidBlock().to(self.device)
        result = net.load_state_dict(test.state_dict())
        self.check_load_state_dict_output(result)
        x = torch.rand(1, 512, 8, 8).to(self.device)
        self.check_output(net, test, x)

        # Decoder
        test = self.model.decoder.mid_block
        net = UNetMidBlock().to(self.device)
        result = net.load_state_dict(test.state_dict())
        self.check_load_state_dict_output(result)
        x = torch.rand(1, 512, 8, 8).to(self.device)
        self.check_output(net, test, x)

    def test_encoder(self):
        test = self.model.encoder
        net = Encoder().to(self.device)
        result = net.load_state_dict(test.state_dict())
        self.check_load_state_dict_output(result)

        x = torch.rand(1,3,16,16).to(self.device)
        self.check_output(net, test, x)

    def test_decoder(self):
        test = self.model.decoder
        net = Decoder().to(self.device)
        result = net.load_state_dict(test.state_dict())
        self.check_load_state_dict_output(result)

        x = torch.rand(1,4,8,8).to(self.device)
        self.check_output(net, test, x)

    def test_autoencoder(self):
        test = self.model
        net = AutoEncoder().to(self.device)
        result = net.load_state_dict(test.state_dict())

        # assert no missing or unexpected keys
        self.assertEqual(len(result[0]), 0, "Missing keys detected.")
        self.assertEqual(len(result[1]), 0, "Unexpected keys detected.")

        # Random input should give the same result
        x = torch.rand(1,3,64,64).to(self.device)
        y1 = test(x).sample
        y2, _, _ = net(x)
        self.assertTrue(torch.allclose(y1, y2, atol=.1), "VAE output should be roughly similar (up to random noise in the embedding)")

if __name__ == '__main__':
    unittest.main()
