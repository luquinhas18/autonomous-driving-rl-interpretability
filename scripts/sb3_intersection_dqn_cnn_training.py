import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from captum.attr import IntegratedGradients, DeepLift, DeepLiftShap, InputXGradient, GuidedBackprop, GradientShap, Saliency
from captum.attr import visualization as viz

import highway_env  # noqa: F401

import torch
import numpy as np

NUM_STEPS = int(1e5)
EXP_DIR_NAME = "intersection_cnn_captum_ig"
ENV = "intersection-v0"

def analyze_model(model):
    """Function to analyze the internal model structure and parameters"""
    print("=" * 50)
    print("MODEL ANALYSIS")
    print("=" * 50)
    
    # Architecture
    print("\n1. ARCHITECTURE:")
    print(f"Policy: {model.policy}")
    print(f"Q-Network: {model.q_net}")
    print(f"Target Q-Network: {model.q_net_target}")
    
    # Parameters
    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"\n2. PARAMETERS:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Layer details
    print(f"\n3. LAYER DETAILS:")
    for name, module in model.policy.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            num_params = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {module} ({num_params:,} params)")
    
    # Configuration
    print(f"\n4. CONFIGURATION:")
    print(f"Learning rate: {model.learning_rate}")
    print(f"Buffer size: {model.buffer_size}")
    print(f"Batch size: {model.batch_size}")
    print(f"Gamma: {model.gamma}")
    print(f"Target update interval: {model.target_update_interval}")
    
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'architecture': str(model.policy),
        'config': {
            'learning_rate': model.learning_rate,
            'buffer_size': model.buffer_size,
            'batch_size': model.batch_size,
            'gamma': model.gamma,
        }
    }


def train_env():
    env = gym.make(
        ENV,
        config={
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (128, 64),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                "scaling": 1.75,
            },
        },
    )
    env.reset()
    return env


def test_env():
    env = gym.make(
        ENV,
        render_mode="rgb_array",
        config={
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (128, 64),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                "scaling": 1.75,
            },
        },
    )
    env.unwrapped.config.update({"policy_frequency": 1, "duration": 30})
    env.reset()
    return env


def inference():
    # Record video
    model = DQN.load(f"{EXP_DIR_NAME}/model")
    
    # Analyze the trained model
    model_info = analyze_model(model)
    print(model_info)

    env = DummyVecEnv([test_env])
    video_length = 2 * env.envs[0].unwrapped.config["duration"]
    print(f"Video length: {video_length}")
    env = VecVideoRecorder(
        env,
        f"{EXP_DIR_NAME}/videos/",
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix="dqn-agent-captum",
    )
    try:
        obs = env.reset()
        for step in range(video_length + 1):
            print(f"Step {step}/{video_length}")
            action, _ = model.predict(obs)
            obs, _, _, _ = env.step(action)
            
            # Get RGB image from the environment
            rgb_image = env.envs[0].render()  # This gives you the RGB array
            print(f"Step {step}: RGB image shape: {rgb_image.shape}")
            # rgb_image is a numpy array with shape (height, width, 3) with values 0-255
    except Exception as e:
        print(f"Error: {e}")
        env.close()



def train():
    model = DQN(
        "CnnPolicy",
        DummyVecEnv([train_env]),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        exploration_fraction=0.7,
        verbose=1,
        tensorboard_log=f"{EXP_DIR_NAME}/",
    )
    print(f"Training model for {NUM_STEPS} steps...")
    model.learn(total_timesteps=NUM_STEPS)
    model.save(f"{EXP_DIR_NAME}/model")
    
    # Analyze the trained model
    model_info = analyze_model(model)


if __name__ == "__main__":
    # Train
    train()

    # Inference
    inference()
