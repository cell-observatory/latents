# Latents: Autoencoder Training

This repository implements various compression techniques (VAE, VQ-VAE) for volumetric data using containerized training environments.

## Features

- Containerized training environment (Docker/Apptainer)
- Distributed training with Ray
- Hyperparameter tuning with Ray Tune
- Mixed precision training (AMP)
- TensorBoard integration for experiment tracking
- Support for multiple platforms:
  - Local machines
  - HPC clusters
  - (TODO) Kubernetes

## Usage

### Local Training

See VSCode development setup below. Or use the terminal:

1. Start the container:
```bash
docker run --network host -u 1000 --privileged -v $(pwd):/app/platform -w /app/platform --env PYTHONUNBUFFERED=1 --pull missing -it --rm  --ipc host --gpus all ghcr.io/cell-observatory/platform:develop_torch_cuda_12_8 bash
```

2. Run training:
```bash
python train_ray.py --batch_size 4 --epochs 100 --num_gpus 8
```

3. Start TensorBoard (in a separate terminal):
```bash
# Inside the container
tensorboard --logdir runs --bind_all
```

Then open your browser at `http://localhost:6006`

### Hyperparameter Tuning

[TODO]

### HPC Cluster Submission

#### SLURM with Apptainer

[TODO]

#### LSF with Apptainer

[TODO]

## Configuration

Training parameters:

- `--batch_size`: Batch size per GPU (default: 4)
- `--epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 1e-5)
- `--kl_weight`: KL divergence weight (default: 0.1)
- `--num_gpus`: Number of GPUs to use (default: all available)
- `--num_cpus`: Number of CPUs to use (default: all available)
- `--tune`: Enable hyperparameter tuning
- `--log_interval`: Logging interval (default: 10)
- `--save_interval`: Checkpoint interval (default: 5)
- `--log_dir`: Directory for TensorBoard logs (default: 'runs')

## Monitoring

Training progress can be monitored through:

1. Console output
2. TensorBoard dashboard (available at http://localhost:6006)
   - Training metrics (loss, reconstruction loss, KL loss)
   - Learning rate
   - Hyperparameter tuning results
3. Ray Dashboard (for distributed training)
4. Checkpoint files in `checkpoints/`

## VS Code Development Setup

### Prerequisites

1. Install VS Code: https://code.visualstudio.com/
2. Install Docker: https://docs.docker.com/get-docker/
3. Install VS Code Extensions:
   - Docker
   - Dev Containers

### Setting up Development Container

1. Create a `.devcontainer` directory in your project root:
```bash
mkdir .devcontainer
```

2. Create a `devcontainer.json` file:
```bash
touch .devcontainer/devcontainer.json
```

3. Add the following configuration to `.devcontainer/devcontainer.json`:
```json
{
    "name": "Autoencoder Development",
    "image": "ghcr.io/cell-observatory/platform:develop_torch_cuda_12_8",
    "customizations": {
        "vscode": {
            "extensions": [
                "ms-python.python",
                "ms-python.vscode-pylance",
                "ms-azuretools.vscode-docker",
                "ms-vscode.remote-containers",
                "eamodio.gitlens"
            ],
            "settings": {
                "python.defaultInterpreterPath": "/usr/local/bin/python",
                "python.linting.enabled": true,
                "python.linting.pylintEnabled": true,
                "python.formatting.provider": "black",
                "editor.formatOnSave": true,
                "editor.rulers": [
                    88
                ],
                "files.trimTrailingWhitespace": true
            }
        }
    },
    "forwardPorts": [
        6006
    ], // TensorBoard port
    "remoteUser": "1000",
    "runArgs": [
        "--network",
        "host",
        "--privileged",
        "--ipc",
        "host",
        "--gpus",
        "all",
        "--pull",
        "missing",
        "--shm-size",
        "8G"
    ],
    "containerEnv": {
        "PYTHONUNBUFFERED": "1"
    }
}
```

### Starting Development Environment

1. Open VS Code
2. Open your project folder
3. When prompted, click "Reopen in Container" or:
   - Press `F1` or `Ctrl+Shift+P`
   - Type "Dev Containers: Reopen in Container"
   - Press Enter

### Using the Development Environment

1. **Terminal Access**:
   - Open a new terminal in VS Code (`Ctrl+` `)
   - The terminal will be inside the container
   - All commands run in the container environment

2. **Running Training**:
```bash
# Start training
python train_ray.py --batch_size 4 --epochs 100 --num_gpus 8

# In a separate terminal, start TensorBoard
tensorboard --logdir runs --bind_all
```

3. **Debugging**:
   - Set breakpoints in your code
   - Use VS Code's debugger (F5)
   - Configure launch.json for specific debugging scenarios

4. **Git Integration**:
   - Git commands work as normal
   - GitLens provides enhanced Git features
   - Credentials are shared with host machine

### Running Tests

The project uses Python's unittest framework for testing. Tests can be run in two ways:

1. Using VS Code Test Explorer:
   - Press `Ctrl+;+A` to run all tests or,
   - Click the play button next to individual tests or at the top to run all tests

2. Using command line:
   ```bash
   # Run all tests
   python -m unittest discover -v -s tests

   # Run a specific test file
   python -m unittest tests/test_autoencoder2d.py -v

   # Run a specific test case
   python -m unittest tests.test_autoencoder2d.TestAutoEncoder2DAgainstDiffusers -v
   ```
