from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention

@dataclass
class ResnetBlockConfig:
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    norm_num_groups: int = 32
    norm_eps: float = 1e-6
    dropout_p: float = 0
    dim: int = 2

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, config: ResnetBlockConfig = None, ):
        super(ResnetBlock, self).__init__()

        if config is None:
            config = ResnetBlockConfig()

        if in_channels % config.norm_num_groups != 0:
            raise ValueError("Number of channels must be divisible by number of groups")

        self.norm1 = nn.GroupNorm(config.norm_num_groups, in_channels, eps=config.norm_eps, affine=True)

        self.shortcut = in_channels != out_channels
        if config.dim == 2:
            self.conv1 = nn.Conv2d(in_channels, out_channels, config.kernel_size, config.stride, config.padding)
            self.conv2 = nn.Conv2d(out_channels, out_channels, config.kernel_size, config.stride, config.padding)
            if self.shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        elif config.dim == 3:
            self.conv1 = nn.Conv3d(in_channels, out_channels, config.kernel_size, config.stride, config.padding)
            self.conv2 = nn.Conv3d(out_channels, out_channels, config.kernel_size, config.stride, config.padding)
            if self.shortcut:
                self.conv_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            raise NotImplementedError(f"{config.dim}D ResnetBlock is not supported.")

        self.norm2 = nn.GroupNorm(config.norm_num_groups, out_channels, eps=config.norm_eps, affine=True)
        self.dropout = nn.Dropout(p=config.dropout_p, inplace=False)
        self.nonlinearity = nn.SiLU()

    def forward(self, x):
        # input shape:
        #   3D: B,Cin,D,H,W
        #   2D: B,Cin,H,W
        h = self.nonlinearity(self.norm1(x))
        h = self.nonlinearity(self.norm2(self.conv1(h)))
        h = self.dropout(h)
        h = self.conv2(h)
        if self.shortcut:
            x = self.conv_shortcut(x)
        # Output shape:
        #   3D: B,Cout,D,H,W
        #   2D: B,Cout,H,W
        return x + h

@dataclass
class DownsampleConfig:
    kernel_size: int = 3
    stride: int = 2
    padding: int = 0
    dim: int = 2

class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, config: Optional[DownsampleConfig] = None):
        super(Downsample, self).__init__()

        if config is None:
            config = DownsampleConfig()

        self.config = config

        if config.dim == 2:
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)  # (left, right, top, bottom)
            self.conv = nn.Conv2d(in_channels, out_channels, config.kernel_size, config.stride, config.padding)
        elif config.dim == 3:
            self.pad = nn.ConstantPad3d((0, 1, 0, 1, 0, 1), 0)  # (left, right, top, bottom, front, back)
            self.conv = nn.Conv3d(in_channels, out_channels, config.kernel_size, config.stride, config.padding)
        else:
            raise NotImplementedError(f"{config.dim}D downsampling is not supported.")

    def forward(self, x):
        x = self.pad(x)
        return self.conv(x)

@dataclass
class UpsampleConfig:
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    dim: int = 2
    scale_factor: float = 2.0
    mode: str = "nearest"

class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, config: Optional[UpsampleConfig] = None):
        super(Upsample, self).__init__()

        if config is None:
            config = UpsampleConfig()

        self.config = config

        if config.dim == 2:
            self.upsample = nn.Upsample(scale_factor=config.scale_factor, mode=config.mode)
            self.conv = nn.Conv2d(in_channels, out_channels, config.kernel_size, config.stride, config.padding)
        elif config.dim == 3:
            self.upsample = nn.Upsample(scale_factor=config.scale_factor, mode=config.mode)
            self.conv = nn.Conv3d(in_channels, out_channels, config.kernel_size, config.stride, config.padding)
        else:
            raise NotImplementedError(f"{config.dim}D upsampling is not supported.")

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)

@dataclass
class DownEncoderBlockConfig:
    resnet_block_config: ResnetBlockConfig = field(default_factory=ResnetBlockConfig)
    downsample_config: DownsampleConfig = field(default_factory=DownsampleConfig)

class DownEncoderBlock(nn.Module):
    def __init__(self, h_channels: List[int], n_downsamplers: int = 1,
                 config: Optional[DownEncoderBlockConfig] = None, ):
        super(DownEncoderBlock, self).__init__()

        if config is None:
            config = DownEncoderBlockConfig()

        self.resnets = nn.ModuleList()
        for i in range(len(h_channels) - 1):
            self.resnets.append(ResnetBlock(h_channels[i], h_channels[i + 1], config.resnet_block_config))

        self.downsamplers = nn.ModuleList()
        for i in range(n_downsamplers):
            self.downsamplers.append(Downsample(h_channels[-1], h_channels[-1], config.downsample_config))

    def forward(self, x):
        for i in range(len(self.resnets)):
            x = self.resnets[i](x)
        for i in range(len(self.downsamplers)):
            x = self.downsamplers[i](x)
        return x

@dataclass
class UpDecoderBlockConfig:
    resnet_block_config: ResnetBlockConfig = field(default_factory=ResnetBlockConfig)
    upsample_config: UpsampleConfig = field(default_factory=UpsampleConfig)

class UpDecoderBlock(nn.Module):
    def __init__(self, h_channels: List[int], n_upsamplers: int = 1, config: Optional[UpDecoderBlockConfig] = None, ):
        super(UpDecoderBlock, self).__init__()
        if config is None:
            config = UpDecoderBlockConfig()

        self.resnets = nn.ModuleList()
        for i in range(len(h_channels) - 1):
            self.resnets.append(ResnetBlock(h_channels[i], h_channels[i + 1], config.resnet_block_config))

        self.upsamplers = nn.ModuleList()
        for i in range(n_upsamplers):
            self.upsamplers.append(Upsample(h_channels[-1], h_channels[-1], config.upsample_config))

    def forward(self, x):
        for i in range(len(self.resnets)):
            x = self.resnets[i](x)
        for i in range(len(self.upsamplers)):
            x = self.upsamplers[i](x)
        return x

@dataclass
class UNetMidBlockConfig:
    resnet_block_config: ResnetBlockConfig = field(default_factory=ResnetBlockConfig)
    n_resnet_blocks: int = 2
    norm_num_groups: int = 32
    norm_eps: float = 1e-6

class UNetMidBlock(nn.Module):
    def __init__(self,
                 n_attentions=1,
                 h_channels=512,
                 config: Optional[UNetMidBlockConfig] = None,
                 ):
        super(UNetMidBlock, self).__init__()

        if config is None:
            config = UNetMidBlockConfig()

        self.attentions = nn.ModuleList()
        for i in range(n_attentions):
            self.attentions.append(
                Attention(
                    h_channels,
                    heads=1,
                    dim_head=h_channels,
                    rescale_output_factor=1,
                    eps=config.norm_eps,
                    norm_num_groups=config.norm_num_groups,
                    spatial_norm_dim=None,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                )
            )

        self.resnets = nn.ModuleList()
        for i in range(config.n_resnet_blocks):
            self.resnets.append(ResnetBlock(h_channels, h_channels, config.resnet_block_config))

    def forward(self, x):
        x = self.resnets[0](x)
        for attention, resnet in zip(self.attentions, self.resnets[1:]):
            if attention is not None:
                x = attention(x)
            if resnet is not None:
                x = resnet(x)
        return x

@dataclass
class EncoderConfig:
    down_encoder_block_config: DownEncoderBlockConfig = field(default_factory=DownEncoderBlockConfig)
    unet_mid_block_config: UNetMidBlockConfig = field(default_factory=UNetMidBlockConfig)
    n_attentions: int = 1
    conv_in_kernel_size: int = 3
    conv_in_stride: int = 1
    conv_in_padding: int = 1
    conv_in_dim: int = 2
    conv_out_kernel_size: int = 3
    conv_out_stride: int = 1
    conv_out_padding: int = 1
    conv_out_dim: int = 2
    norm_num_groups: int = 32
    norm_eps: float = 1e-6

class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 8,
                 h_channels: Tuple[List[int], ...] = (
                         [128, 128, 128],
                         [128, 256, 256],
                         [256, 512, 512],
                         [512, 512, 512]
                 ),
                 n_downsamplers: Tuple[int] = (1, 1, 1, 0),
                 config: Optional[EncoderConfig] = None
                 ):
        super(Encoder, self).__init__()

        if config is None:
            config = EncoderConfig()

        if config.conv_in_dim == 2:
            self.conv_in = nn.Conv2d(
                in_channels,
                h_channels[0][0],
                kernel_size=config.conv_in_kernel_size,
                stride=config.conv_in_stride,
                padding=config.conv_in_padding
            )
        elif config.conv_in_dim == 3:
            self.conv_in = nn.Conv3d(
                in_channels,
                h_channels[0][0],
                kernel_size=config.conv_in_kernel_size,
                stride=config.conv_in_stride,
                padding=config.conv_in_padding
            )
        else:
            raise NotImplementedError(f"{config.conv_in_dim}D conv_in is not supported")

        self.down_blocks = nn.ModuleList()
        for i in range(len(h_channels)):
            self.down_blocks.append(DownEncoderBlock(
                h_channels[i],
                n_downsamplers=n_downsamplers[i],
                config=config.down_encoder_block_config
            ))

        self.mid_block = UNetMidBlock(
            n_attentions=config.n_attentions,
            h_channels=h_channels[-1][-1],
            config=config.unet_mid_block_config
        )

        self.conv_norm_out = nn.GroupNorm(config.norm_num_groups, h_channels[-1][-1], eps=config.norm_eps, affine=True)
        self.conv_act = nn.SiLU()

        if config.conv_out_dim == 2:
            self.conv_out = nn.Conv2d(
                h_channels[-1][-1],
                out_channels,
                kernel_size=config.conv_out_kernel_size,
                stride=config.conv_out_stride,
                padding=config.conv_out_padding
            )
        elif config.conv_out_dim == 3:
            self.conv_out = nn.Conv3d(
                h_channels[-1][-1],
                out_channels,
                kernel_size=config.conv_out_kernel_size,
                stride=config.conv_out_stride,
                padding=config.conv_out_padding
            )
        else:
            raise NotImplementedError(f"{config.conv_out_dim}D conv_in is not supported")

    def forward(self, x):
        x = self.conv_in(x)
        for down_block in self.down_blocks:
            x = down_block(x)
        x = self.mid_block(x)
        x = self.conv_act(self.conv_norm_out(x))
        x = self.conv_out(x)
        return x

@dataclass
class DecoderConfig:
    updecoder_block_config: UpDecoderBlockConfig = field(default_factory=UpDecoderBlockConfig)
    unet_mid_block_config: UNetMidBlockConfig = field(default_factory=UNetMidBlockConfig)
    n_attentions: int = 1
    conv_in_kernel_size: int = 3
    conv_in_stride: int = 1
    conv_in_padding: int = 1
    conv_in_dim: int = 2
    conv_out_kernel_size: int = 3
    conv_out_stride: int = 1
    conv_out_padding: int = 1
    conv_out_dim: int = 2
    norm_num_groups: int = 32
    norm_eps: float = 1e-6

class Decoder(nn.Module):
    def __init__(self,
                 in_channels: int = 4,
                 out_channels: int = 3,
                 h_channels: Tuple[List[int], ...] = (
                         [512, 512, 512, 512],
                         [512, 512, 512, 512],
                         [512, 256, 256, 256],
                         [256, 128, 128, 128]
                 ),
                 n_upsamplers: Tuple[int] = (1, 1, 1, 0),
                 config: DecoderConfig = None
                 ):
        super(Decoder, self).__init__()

        if config is None:
            config = DecoderConfig()

        if config.conv_in_dim == 2:
            self.conv_in = nn.Conv2d(
                in_channels,
                h_channels[0][0],
                kernel_size=config.conv_in_kernel_size,
                stride=config.conv_in_stride,
                padding=config.conv_in_padding
            )
        elif config.conv_in_dim == 3:
            self.conv_in = nn.Conv3d(
                in_channels,
                h_channels[0][0],
                kernel_size=config.conv_in_kernel_size,
                stride=config.conv_in_stride,
                padding=config.conv_in_padding
            )
        else:
            raise NotImplementedError(f"{config.conv_in_dim}D conv_in is not supported")

        self.up_blocks = nn.ModuleList()
        for i in range(len(h_channels)):
            self.up_blocks.append(
                UpDecoderBlock(h_channels[i], n_upsamplers=n_upsamplers[i], config=config.updecoder_block_config))

        self.mid_block = UNetMidBlock(
            n_attentions=config.n_attentions,
            h_channels=h_channels[0][0],
            config=config.unet_mid_block_config
        )

        self.conv_norm_out = nn.GroupNorm(config.norm_num_groups, h_channels[-1][-1], eps=config.norm_eps, affine=True)
        self.conv_act = nn.SiLU()

        if config.conv_out_dim == 2:
            self.conv_out = nn.Conv2d(
                h_channels[-1][-1],
                out_channels,
                kernel_size=config.conv_out_kernel_size,
                stride=config.conv_out_stride,
                padding=config.conv_out_padding
            )
        elif config.conv_out_dim == 3:
            self.conv_out = nn.Conv3d(
                h_channels[-1][-1],
                out_channels,
                kernel_size=config.conv_out_kernel_size,
                stride=config.conv_out_stride,
                padding=config.conv_out_padding
            )
        else:
            raise NotImplementedError(f"{config.conv_out_dim}D conv_out is not supported")

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        x = self.conv_act(self.conv_norm_out(x))
        x = self.conv_out(x)
        return x

@dataclass
class AutoEncoderConfig:
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    decoder_config: DecoderConfig = field(default_factory=DecoderConfig)
    conv_in_kernel_size: int = 3
    conv_in_stride: int = 1
    conv_in_padding: int = 1
    conv_out_kernel_size: int = 3
    conv_out_stride: int = 1
    conv_out_padding: int = 1
    quant_conv_in_channels: int = 8
    quant_conv_out_channels: int = 8
    quant_conv_padding: int = 0
    quant_conv_dim: int = 2
    post_quant_conv_in_channels: int = 4
    post_quant_conv_out_channels: int = 4
    post_quant_conv_padding: int = 0
    post_quant_conv_dim: int = 2

class AutoEncoder(nn.Module):
    def __init__(self,
                 encoder_in_channels=3,
                 encoder_out_channels=8,
                 encoder_h_channels=(
                         [128, 128, 128],
                         [128, 256, 256],
                         [256, 512, 512],
                         [512, 512, 512]
                 ),
                 n_downsamplers=(1, 1, 1, 0),
                 decoder_in_channels=4,
                 decoder_out_channels=3,
                 decoder_h_channels=(
                         [512, 512, 512, 512],
                         [512, 512, 512, 512],
                         [512, 256, 256, 256],
                         [256, 128, 128, 128]
                 ),
                 n_upsamplers=(1, 1, 1, 0),
                 config: AutoEncoderConfig = None
                 ):
        super(AutoEncoder, self).__init__()

        if config is None:
            config = AutoEncoderConfig()

        self.encoder = Encoder(
            in_channels=encoder_in_channels,
            out_channels=encoder_out_channels,
            h_channels=encoder_h_channels,
            n_downsamplers=n_downsamplers,
            config=config.encoder_config,
        )

        assert config.quant_conv_out_channels == 2 * decoder_in_channels, "quant_conv_out"

        self.decoder = Decoder(
            in_channels=decoder_in_channels,
            out_channels=decoder_out_channels,
            h_channels=decoder_h_channels,
            n_upsamplers=n_upsamplers,
            config=config.decoder_config
        )

        if config.quant_conv_dim == 2:
            self.quant_conv = nn.Conv2d(config.quant_conv_in_channels,
                                        config.quant_conv_out_channels,
                                        kernel_size=1, stride=1,
                                        padding=config.quant_conv_padding)
        elif config.quant_conv_dim == 3:
            self.quant_conv = nn.Conv3d(config.quant_conv_in_channels,
                                        config.quant_conv_out_channels,
                                        kernel_size=1, stride=1,
                                        padding=config.quant_conv_padding)
        else:
            raise NotImplementedError(f"{config.quant_conv_dim}D quant_conv is not supported")

        if config.post_quant_conv_dim == 2:
            self.post_quant_conv = nn.Conv2d(config.post_quant_conv_in_channels,
                                             config.post_quant_conv_out_channels,
                                             kernel_size=1, stride=1,
                                             padding=config.post_quant_conv_padding)
        elif config.post_quant_conv_dim == 3:
            self.post_quant_conv = nn.Conv3d(config.post_quant_conv_in_channels,
                                             config.post_quant_conv_out_channels,
                                             kernel_size=1, stride=1,
                                             padding=config.post_quant_conv_padding)
        else:
            raise NotImplementedError(f"{config.post_quant_conv_dim}D post_quant_conv is not supported")

    def forward(self, x):
        x = self.encode(x)
        z, mean, logvar = self.reparameterize(x)
        x = self.decode(z)
        return x, mean, logvar

    def encode(self, x):
        x = self.encoder(x)
        return self.quant_conv(x)

    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def reparameterize(self, x):
        mean, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z, mean, logvar