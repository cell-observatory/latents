import os
import sys
import time
import logging
import argparse
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

import ray
from ray import train
from ray.train import Checkpoint, ScalingConfig
from ray.train.torch import TorchTrainer, prepare_model, prepare_data_loader
from ray.tune import TuneConfig, Tuner
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.logger.tensorboardx import TBXLoggerCallback

from models.autoencoder import (
    AutoEncoder, AutoEncoderConfig, ResnetBlockConfig, DownsampleConfig, UpsampleConfig,
    UNetMidBlockConfig, UpDecoderBlockConfig, DownEncoderBlockConfig, EncoderConfig,
    DecoderConfig
)
from data.membrane_histone_dataset import MembraneHistoneDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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

def train_func(config: Dict[str, Any]):
    """Training function for Ray Train."""
    torch.set_float32_matmul_precision("high")

    # Set multiprocessing start method to spawn
    mp.set_start_method('spawn', force=True)

    # Initialize TensorBoard
    log_dir = os.path.join("runs", f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir=log_dir)

    # Create model
    model_config = get_model_config()
    model = AutoEncoder(
        encoder_in_channels=2,
        encoder_out_channels=8,
        decoder_in_channels=4,
        decoder_out_channels=2,
        config=model_config
    )

    # Prepare model for distributed training
    model = prepare_model(model)

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"]
    )

    # Create dataset and dataloader
    dataset = MembraneHistoneDataset(
        data_path=config["data_path"],
        time_window=config["time_window"],
        crop_size=config["crop_size"]
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        multiprocessing_context='spawn'  # Explicitly use spawn
    )
    dataloader = prepare_data_loader(dataloader)

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()

    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0

        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                output, mean, logvar = model(data)
                # Reconstruction loss
                recon_loss = nn.MSELoss()(output, data)
                # KL divergence loss
                kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
                # Total loss
                loss = recon_loss + config["kl_weight"] * kl_loss

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if batch_idx % config["log_interval"] == 0:
                metrics = {
                    "loss": loss.item(),
                    "recon_loss": recon_loss.item(),
                    "kl_loss": kl_loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "batch": batch_idx
                }
                # Log to TensorBoard
                for name, value in metrics.items():
                    writer.add_scalar(f"train/{name}", value, epoch * len(dataloader) + batch_idx)
                train.report(metrics)

        # Update learning rate
        scheduler.step()

        # Calculate epoch metrics
        avg_loss = total_loss / len(dataloader)
        epoch_metrics = {
            "epoch": epoch,
            "avg_loss": avg_loss
        }
        # Log epoch metrics to TensorBoard
        writer.add_scalar("epoch/avg_loss", avg_loss, epoch)
        train.report(epoch_metrics)

        # Save checkpoint
        if (epoch + 1) % config["save_interval"] == 0:
            checkpoint = Checkpoint.from_dict({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
            })
            train.report({"avg_loss": avg_loss}, checkpoint=checkpoint)

    writer.close()

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

    # Ray parameters
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--num_gpus', type=int, default=-1, help='Number of GPUs to use')
    parser.add_argument('--num_cpus', type=int, default=-1, help='Number of CPUs to use')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')

    # Logging and checkpointing
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=5, help='Checkpoint saving interval')
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory for TensorBoard logs')

    args = parser.parse_args()

    # Initialize Ray with explicit resource configuration
    if args.num_gpus == -1:
        args.num_gpus = torch.cuda.device_count()

    # Set training mode based on GPU availability
    use_gpu = args.num_gpus > 0 and torch.cuda.is_available()
    if not use_gpu:
        logger.info("No GPUs available. Running in CPU-only mode.")
        args.num_gpus = 1  # Use 1 worker for CPU training
        args.num_cpus = max(1, args.num_cpus)  # Ensure at least 1 CPU

    # Initialize Ray with explicit resource configuration
    ray.init(
        num_cpus=args.num_cpus if args.num_cpus > 0 else None,
        num_gpus=args.num_gpus if use_gpu else 0,
        local_mode=args.num_gpus == 0,  # Use local mode for CPU-only training
        ignore_reinit_error=True,
        include_dashboard=True,
        dashboard_host="0.0.0.0",  # Allow external access to dashboard
        dashboard_port=8265,  # Default Ray dashboard port
    )

    logger.info(f"Ray initialized with {args.num_cpus if args.num_cpus > 0 else 'all'} CPUs and {args.num_gpus if use_gpu else 0} GPUs")

    # Create checkpoint and log directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Base training configuration
    train_config = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "kl_weight": args.kl_weight,
        "num_workers": args.num_workers,
        "log_interval": args.log_interval,
        "save_interval": args.save_interval,
        "data_path": args.data_path,
        "time_window": args.time_window,
        "crop_size": args.crop_size,
        "log_dir": args.log_dir,
    }

    if args.tune:
        # Hyperparameter tuning configuration
        param_space = {
            "learning_rate": ray.tune.loguniform(1e-5, 1e-3),
            "weight_decay": ray.tune.loguniform(1e-6, 1e-4),
            "kl_weight": ray.tune.uniform(0.01, 0.5),
            "batch_size": ray.tune.choice([2, 4, 8]),
        }
        # Update train_config with search space
        train_config.update(param_space)

        # Configure the tuner
        tuner = Tuner(
            train_func,
            tune_config=TuneConfig(
                metric="avg_loss",
                mode="min",
                num_samples=20,  # Number of trials
                scheduler=ASHAScheduler(
                    time_attr="epoch",
                    metric="avg_loss",
                    mode="min",
                    max_t=args.epochs,
                    grace_period=10,
                ),
                search_alg=BayesOptSearch(
                    metric="avg_loss",
                    mode="min",
                ),
            ),
            run_config=ray.air.RunConfig(
                name="3d_autoencoder_tuning",
                callbacks=[TBXLoggerCallback()],
                checkpoint_config=ray.air.CheckpointConfig(
                    checkpoint_score_attribute="avg_loss",
                    num_to_keep=2,
                ),
            ),
            param_space=train_config,
            scaling_config=ScalingConfig(
                num_workers=args.num_gpus,
                use_gpu=use_gpu,
                resources_per_worker={
                    "GPU": 1 if use_gpu else 0,
                    "CPU": args.num_cpus
                },
            ),
        )

        # Run the tuning
        results = tuner.fit()
        best_result = results.get_best_result(metric="avg_loss", mode="min")
        logger.info(f"Best trial config: {best_result.config}")
        logger.info(f"Best trial final validation loss: {best_result.metrics['avg_loss']}")

    else:
        # Single training run
        trainer = TorchTrainer(
            train_func,
            train_loop_config=train_config,
            scaling_config=ScalingConfig(
                num_workers=args.num_gpus,
                use_gpu=use_gpu,
                resources_per_worker={
                    "GPU": 1 if use_gpu else 0,
                    "CPU": args.num_cpus
                },
            ),
            run_config=ray.air.RunConfig(
                name="3d_autoencoder",
                callbacks=[TBXLoggerCallback()],
                checkpoint_config=ray.air.CheckpointConfig(
                    checkpoint_score_attribute="avg_loss",
                    num_to_keep=2,
                ),
            ),
        )

        # Run the training
        result = trainer.fit()
        logger.info(f"Final validation loss: {result.metrics['avg_loss']}")

    # Shutdown Ray
    ray.shutdown()

if __name__ == '__main__':
    main()