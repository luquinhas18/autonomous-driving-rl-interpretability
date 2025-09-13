# Interpretability in Autonomous Driving: Visual Attribution Analysis of RL Agents

This repository implements visual attribution analysis for reinforcement learning agents trained on autonomous driving tasks using the Highway Environment. The project uses Stable-Baselines3 for RL training and Captum for interpretability analysis with Integrated Gradients.

## Description

This project focuses on understanding how deep reinforcement learning agents make decisions in autonomous driving scenarios. By applying attribution methods like Integrated Gradients, we can visualize which parts of the input observations (grayscale frames) are most important for the agent's decision-making process.

### Key Features

- **RL Training**: DQN agents trained on highway and intersection driving scenarios
- **Visual Attribution**: Integrated Gradients analysis to understand agent decisions
- **Frame-wise Analysis**: Attribution analysis across stacked observation frames
- **Video Generation**: Automated creation of attribution visualization videos
- **Multiple Environments**: Support for both highway and intersection driving scenarios

## Setup

### Repository Clone

```bash
git clone <repository-url>
cd autonomous-driving-rl-interpretability
```

### Dependencies

The main dependencies are:

- **PyTorch**: Deep learning framework
- **Captum**: Model interpretability library
- **Highway-Env**: Autonomous driving simulation environment
- **Stable-Baselines3**: Reinforcement learning algorithms
- **OpenCV**: Video processing
- **Matplotlib**: Visualization

### Installation

#### Option 1: Conda Environment (Recommended)

```bash
# Create conda environment
conda create -n autonomous-driving python=3.9

# Activate environment
conda activate autonomous-driving

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install captum
pip install highway-env
pip install stable-baselines3[extra]
pip install opencv-python
pip install matplotlib
pip install gymnasium
pip install pillow
```

#### Option 2: pip Installation

```bash
pip install torch torchvision torchaudio
pip install captum
pip install highway-env
pip install stable-baselines3[extra]
pip install opencv-python
pip install matplotlib
pip install gymnasium
pip install pillow
```

## Usage

### Training RL Agents

#### Highway Environment Training

```bash
python scripts/sb3_highway_dqn_cnn_training.py
```

This script trains a DQN agent on the highway environment with:
- CNN policy for processing grayscale observations
- 4-frame stacked observations (128x64 pixels)
- 100,000 training steps
- Model saved to `highway_cnn_captum_ig/model/`

#### Intersection Environment Training

```bash
python scripts/sb3_intersection_dqn_cnn_training.py
```

This script trains a DQN agent on the intersection environment with similar configuration.

### Attribution Analysis

#### Comprehensive Attribution Analysis

```bash
python scripts/captum_attribution_analysis.py
```

This script performs detailed attribution analysis including:
- Frame-wise attribution analysis
- Comprehensive visualizations

This creates:
- Attribution visualizations for each frame
- RGB overlay visualizations

### Video Generation

Create videos from attribution visualizations:

```bash
python scripts/video_writer.py --input_dir ./intersection_cnn_simple_captum_ig --output_dir ./videos --type both
```

Options:
- `--type attribution`: Attribution videos only
- `--type rgb`: RGB overlay videos only  
- `--type both`: Both types of videos
- `--type side_by_side`: Combined side-by-side video
- `--fps 10`: Adjust frames per second

## Project Structure

```
autonomous-driving-rl-interpretability/
├── scripts/
│   ├── sb3_highway_dqn_cnn_training.py      # Highway environment training
│   ├── sb3_intersection_dqn_cnn_training.py # Intersection environment training
│   ├── captum_attribution_analysis.py       # Comprehensive attribution analysis
│   └── video_writer.py                      # Video generation from visualizations
├── videos/                                  # Generated video outputs
└── README.md
```

## Output Files

### Training Outputs
- `highway_cnn_captum_ig/model/`: Trained DQN model
- `intersection_cnn_captum_ig/model/`: Trained DQN model
- TensorBoard logs for training monitoring

### Attribution Analysis Outputs
- `intersection_cnn_simple_captum_ig/`: Attribution visualization images
- `episode_attribution_analysis.png`: Episode summary plots
- Frame-wise attribution maps and RGB overlays

### [Video Outputs](https://drive.google.com/drive/folders/1F78om50xGEFokI1EcEUmmWazoyRvZZzo?usp=drive_link)


## Key Parameters

### Training Configuration
- **Learning Rate**: 5e-4
- **Buffer Size**: 15,000
- **Batch Size**: 32
- **Gamma**: 0.8
- **Training Steps**: 100,000

### Environment Configuration
- **Observation**: Grayscale, 128x64 pixels
- **Stack Size**: 4 frames
- **Scaling**: 1.75x

### Attribution Configuration
- **Method**: Integrated Gradients
- **Steps**: 50
- **Baseline**: Zero baseline
- **Target**: Chosen action Q-value

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{autonomous-driving-rl-interpretability,
  title={Interpretability in Autonomous Driving: Visual Attribution Analysis of RL Agents},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/autonomous-driving-rl-interpretability}
}
```
