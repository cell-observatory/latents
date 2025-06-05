import os
import sys
import time
import logging
import argparse
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from models.autoencoder import (
    AutoEncoder, AutoEncoderConfig, ResnetBlockConfig, DownsampleConfig, UpsampleConfig,
    UNetMidBlockConfig, UpDecoderBlockConfig, DownEncoderBlockConfig, EncoderConfig,
    DecoderConfig
)
from data.membrane_histone_dataset import MembraneHistoneDataset
from data.crop_transforms import Random4DCrop

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_dataloader(config: Dict[str, Any]):
    """Get dataloader for training."""
    transform = Random4DCrop(time_window=config["time_window"],
                             crop_size=config["crop_size"])
    dataset = MembraneHistoneDataset(
        data_path=config["data_path"],
        transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )
    return dataloader

def get_model_config():
    """Get model configuration for 3D autoencoder."""
    resnet_block_config = ResnetBlockConfig(dim=3, norm_num_groups=32)
    downsample_config = DownsampleConfig(dim=3)
    upsample_config = UpsampleConfig(dim=3)
    unet_mid_block_config = UNetMidBlockConfig(resnet_block_config=resnet_block_config)

    up_decoder_block_config = UpDecoderBlockConfig(
        resnet_block_config=resnet_block_config,
        upsample_config=upsample_config
    )

    down_encoder_block_config = DownEncoderBlockConfig(
        resnet_block_config=resnet_block_config,
        downsample_config=downsample_config
    )

    encoder_config = EncoderConfig(
        down_encoder_block_config=down_encoder_block_config,
        unet_mid_block_config=unet_mid_block_config,
        conv_in_dim=3,
        conv_out_dim=3,
        n_attentions=1
    )

    decoder_config = DecoderConfig(
        up_decoder_block_config=up_decoder_block_config,
        unet_mid_block_config=unet_mid_block_config,
        conv_in_dim=3,
        conv_out_dim=3,
        n_attentions=1
    )

    autoencoder_config = AutoEncoderConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        quant_conv_dim=3,
        post_quant_conv_dim=3,
        quant_conv_in_channels=8,
        quant_conv_out_channels=8,
        post_quant_conv_in_channels=4,
        post_quant_conv_out_channels=4
    )

    return autoencoder_config

def loss_fn(output, data, mean, logvar, kl_weight):
    recon_loss = nn.MSELoss()(output, data)
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss

def train_func(config: Dict[str, Any]):

    torch.set_float32_matmul_precision("high")

    dataloader = get_dataloader(config)

    # Create model
    model_config = get_model_config()
    model = AutoEncoder(
        encoder_in_channels=2,
        encoder_out_channels=8,
        encoder_h_channels=(
            [64, 64, 64],
            [64, 128, 128],
            [128, 256, 256],
            [256, 256, 256]
        ),
        n_downsamplers=(1, 1, 1, 0),
        decoder_in_channels=4,
        decoder_out_channels=2,
        decoder_h_channels=(
            [256, 256, 256, 256],
            [256, 256, 256, 256],
            [256, 128, 128, 128],
            [128, 64, 64, 64]
        ),
        n_upsamplers=(1, 1, 1, 0),
        config=model_config
    )
    model = model.to(config["device"])
    # model = model.to(dtype=torch.bfloat16)

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=config["betas"],
        weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"]
    )

    # Initialize gradient scaler for mixed precision training
    if config["device"] == "cuda":
        scaler = GradScaler(config["device"], enabled = True)

    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0

        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()
            data = data.to(config["device"])

            output, mean, logvar = model(data)
            loss = loss_fn(output, data, mean, logvar, config["kl_weight"])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % config["log_interval"] == 0:
                metrics = {
                    "loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "batch": batch_idx
                }
                print(metrics)

        # Update learning rate
        scheduler.step()

        # Calculate epoch metrics
        # benchmarks here
        # model.eval()
        avg_loss = total_loss / len(dataloader)
        epoch_metrics = {
            "epoch": epoch,
            "avg_loss": avg_loss
        }
        print(epoch_metrics)

        # Save checkpoint
        if (epoch + 1) % config["save_interval"] == 0:
            torch.save(model, f"checkpoints/model_{epoch}_loss_{avg_loss:.4f}.pth")

def main():
    parser = argparse.ArgumentParser(description='Train 3D Autoencoder with Ray')

    # Data parameters
    parser.add_argument('--data_path', type=str, default=r'/CellObservatoryData/20250324_mem_histone', help='Path to the dataset')
    parser.add_argument('--time_window', type=int, default=1, help='Time window for the dataset')
    parser.add_argument('--crop_size', type=tuple, default=(128, 128, 128), help='Crop size for the dataset')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--kl_weight', type=float, default=0.1, help='KL divergence weight')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='Adam optimizer betas')

    # Logging and checkpointing
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=5, help='Checkpoint saving interval')
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory for TensorBoard logs')

    args = parser.parse_args()

    # Create config dictionary from args
    config = {
        # Data parameters
        "data_path": args.data_path,
        "time_window": args.time_window,
        "crop_size": args.crop_size,

        # Training parameters
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "kl_weight": args.kl_weight,
        "betas": args.betas,

        # Device configuration
        "device": "cuda" if torch.cuda.is_available() else "cpu",

        # Logging and checkpointing
        "log_interval": args.log_interval,
        "save_interval": args.save_interval,
        "log_dir": args.log_dir,
    }
    
    train_func(config)

if __name__ == '__main__':
    main()