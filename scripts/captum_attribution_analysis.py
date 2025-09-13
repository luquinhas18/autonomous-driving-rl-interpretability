import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from captum.attr import IntegratedGradients
import highway_env  # noqa: F401
import torch
import numpy as np
import matplotlib.pyplot as plt

# EXP_DIR = "highway_cnn_simple_captum_ig"
# EXP_DIR_NAME = "highway_cnn_captum_ig"
SAVE_INTERVAL = 2

EXP_DIR = "intersection_cnn_simple_captum_ig" # for attribution results.
EXP_DIR_NAME = "intersection_cnn_captum_ig"
ENV = "intersection-v0"

def train_env():
    env = gym.make(
        ENV,
        config={
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (128, 64),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140],
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
                "weights": [0.2989, 0.5870, 0.1140],
                "scaling": 1.75,
            },
        },
    )
    env.unwrapped.config.update({"policy_frequency": 1, "duration": 30})
    env.reset()
    return env


def get_all_frame_attributions(obs_tensor, policy, chosen_action):
    """
    Get attributions for all frames at once, then extract specific frames
    """
    def forward_func(input_tensor):
        # Get Q-values for all actions
        q_values = policy(input_tensor)
        return q_values  # Shape: [batch_size, num_actions]
    
    # Compute attributions for all frames
    ig = IntegratedGradients(forward_func)
    
    # Create baseline on the same device as obs_tensor
    device = obs_tensor.device
    baseline = torch.zeros_like(obs_tensor).to(device)
    
    attributions, delta = ig.attribute(
        obs_tensor,   # shape [1, n_stack, H, W]
        baselines=baseline,
        target=chosen_action,  # target is the chosen action index
        n_steps=50,
        return_convergence_delta=True
    )
    
    # Return attributions for all frames and convergence delta
    return attributions, delta


def get_frame_attribution(obs_tensor, policy, chosen_action, frame_idx):
    """
    Get attribution for a specific frame using the efficient approach
    """
    # Get attributions for all frames at once
    attributions, delta = get_all_frame_attributions(obs_tensor, policy, chosen_action)
    
    # Extract the specific frame we want
    attr_frame = attributions[0, frame_idx].detach().cpu().numpy()  # (H, W)
    
    return attr_frame, delta




def visualize_frame_attribution(obs_tensor, attributions, chosen_action, step, rgb_frame=None, save_path=None):
    """
    Visualize attribution for all frames with optional RGB overlay
    """
    
    if rgb_frame is not None:
        # Create 3 rows: RGB, Grayscale, Attribution
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        print("rgb_frame shape: ", rgb_frame.shape)
        
        # Resize RGB frame to match grayscale observation shape
        from PIL import Image
        gray_shape = obs_tensor[0, 0].shape  # (128, 64)
        
        # First transpose RGB frame from (150, 600, 3) to (600, 150, 3)
        rgb_transposed = rgb_frame.transpose(1, 0, 2)  # (600, 150, 3)
        
        # Then resize to match grayscale shape
        rgb_resized = Image.fromarray(rgb_transposed).resize((gray_shape[1], gray_shape[0]))  # (width, height)
        rgb_resized = np.array(rgb_frame)
        
        # RGB frame (same for all columns since it's the current timestep)
        for frame_idx in range(4):
            axes[0, frame_idx].imshow(rgb_resized)
            axes[0, frame_idx].set_title(f'RGB Frame (Current)')
            axes[0, frame_idx].axis('off')
        
        # Grayscale frames
        for frame_idx in range(4):
            original_frame = obs_tensor[0, frame_idx, :, :].detach().cpu().numpy()
            axes[1, frame_idx].imshow(original_frame, cmap='gray')
            axes[1, frame_idx].set_title(f'Grayscale Frame {frame_idx}')
            axes[1, frame_idx].axis('off')
        
        # Attribution maps
        for frame_idx in range(4):
            attr_map = attributions[frame_idx]
            im = axes[2, frame_idx].imshow(attr_map, cmap='RdBu_r', alpha=0.8)
            axes[2, frame_idx].set_title(f'Attribution Frame {frame_idx}')
            axes[2, frame_idx].axis('off')
            plt.colorbar(im, ax=axes[2, frame_idx])
        
        plt.suptitle(f'Step {step} - Action {chosen_action} - RGB + Grayscale + Attribution', fontsize=16)
        
    else:
        # Original 2-row layout (grayscale only)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for frame_idx in range(4):
            # Original frame
            original_frame = obs_tensor[0, frame_idx, :, :].detach().cpu().numpy()
            axes[0, frame_idx].imshow(original_frame, cmap='gray')
            axes[0, frame_idx].set_title(f'Frame {frame_idx} (Original)')
            axes[0, frame_idx].axis('off')
            
            # Attribution map
            attr_map = attributions[frame_idx]
            im = axes[1, frame_idx].imshow(attr_map, cmap='RdBu_r', alpha=0.8)
            axes[1, frame_idx].set_title(f'Frame {frame_idx} (Attribution)')
            axes[1, frame_idx].axis('off')
            plt.colorbar(im, ax=axes[1, frame_idx])
        
        plt.suptitle(f'Step {step} - Action {chosen_action} - Frame Attributions', fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_rgb_attribution_overlay(obs_tensor, attributions, chosen_action, step, rgb_frame, save_path=None):
    """
    Create RGB overlay visualization showing attribution on top of the actual RGB frame
    """
    from PIL import Image
    import numpy as np
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Get the shape of the grayscale observation
    gray_shape = attributions[0].shape  # (128, 64)
    rgb_shape = rgb_frame.shape  # (150, 600, 3)
    
    print(f"RGB frame shape: {rgb_shape}, Grayscale shape: {gray_shape}")
    
    # First transpose RGB frame from (150, 600, 3) to (600, 150, 3)
    rgb_transposed = rgb_frame.transpose(1, 0, 2)  # (600, 150, 3)
    
    # Then resize to match grayscale observation shape
    rgb_resized = Image.fromarray(rgb_transposed).resize((gray_shape[1], gray_shape[0]))  # (width, height)
    rgb_resized = np.array(rgb_resized)  # Convert back to numpy array
    
    for frame_idx in range(4):
        # RGB frame with attribution overlay
        axes[0, frame_idx].imshow(rgb_resized)
        
        # Overlay attribution for this frame
        attr_map = attributions[frame_idx]
        
        # Normalize attribution to 0-1 for better visualization
        attr_normalized = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)
        
        # Create overlay (red for positive, blue for negative)
        overlay = np.zeros((*attr_map.shape, 4))  # RGBA
        overlay[:, :, 0] = np.maximum(0, attr_normalized)  # Red channel for positive
        overlay[:, :, 2] = np.maximum(0, -attr_normalized + 1)  # Blue channel for negative
        overlay[:, :, 3] = 0.6  # Alpha for transparency
        
        axes[0, frame_idx].imshow(overlay)
        axes[0, frame_idx].set_title(f'RGB + Attribution Frame {frame_idx}')
        axes[0, frame_idx].axis('off')
        
        # Pure attribution map
        im = axes[1, frame_idx].imshow(attr_map, cmap='RdBu_r', alpha=0.8)
        axes[1, frame_idx].set_title(f'Pure Attribution Frame {frame_idx}')
        axes[1, frame_idx].axis('off')
        plt.colorbar(im, ax=axes[1, frame_idx])
    
    plt.suptitle(f'Step {step} - Action {chosen_action} - RGB Attribution Overlay', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def simple_test_plot(attributions, obs_tensor, n_stack):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(f"{EXP_DIR}/simple_plot", exist_ok=True)

    fig, axes = plt.subplots(1, n_stack, figsize=(4*n_stack, 4))
    if n_stack == 1:
        axes = [axes]  # Make it iterable for single subplot
    
    for i in range(n_stack):
        # Extract the specific frame from the tensor
        # obs_tensor shape: [1, 4, 128, 64], so obs_tensor[0, i] gives [128, 64]
        obs_frame = obs_tensor[0, i].detach().cpu().numpy()  # Shape: (128, 64)
        attr_frame = attributions[i]  # Shape: (128, 64)
        
        axes[i].imshow(obs_frame, cmap="gray")
        axes[i].imshow(attr_frame, cmap="hot", alpha=0.5)
        axes[i].set_title(f'Frame {i}')
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig(f"{EXP_DIR}/simple_plot/simple_test_plot.png", dpi=150, bbox_inches='tight')
    plt.show()

    # # Normalize
    # attr_norm = (attr_frame - attr_frame.min()) / (attr_frame.max() - attr_frame.min() + 1e-8)

    # plt.subplot(1, 2, 1)
    # plt.title("Original Frame")
    # plt.imshow(obs_frame, cmap="gray")

    # plt.subplot(1, 2, 2)
    # plt.title("Integrated Gradients Attribution")
    # plt.imshow(obs_frame, cmap="gray")
    # plt.imshow(attr_norm, cmap="hot", alpha=0.5)
    # plt.show()

def simple_attribution_analysis():
    """
    Simple attribution analysis for policy's chosen actions
    """
    import os
    
    # Create experiment directory
    os.makedirs(EXP_DIR, exist_ok=True)
    
    # Load model
    model = DQN.load(f"{EXP_DIR_NAME}/model")
    policy = model.q_net
    
    # Create environment
    env = DummyVecEnv([test_env])
    video_length = 2 * env.envs[0].unwrapped.config["duration"]
    
    obs = env.reset()
    
    print("=" * 50)
    print("SIMPLE FRAME ATTRIBUTION ANALYSIS")
    print("=" * 50)
    print(f"Running for {video_length + 1} steps...")
    
    for step in range(video_length + 1):
        print(f"\nStep {step}/{video_length}")
        print("-" * 30)
        
        # Convert observation to tensor
        obs_tensor = torch.tensor(obs).float().unsqueeze(0)  # Shape: [1, 4, 128, 64]
        
        # Debug: Print tensor shape
        print(f"Observation tensor shape: {obs_tensor.shape}")
        
        # Ensure correct shape for CNN: [batch_size, channels, height, width]
        if len(obs_tensor.shape) == 5:  # [1, 1, 4, 128, 64] - has extra dimension
            obs_tensor = obs_tensor.squeeze(1)  # Remove extra dimension: [1, 4, 128, 64]
            print(f"Fixed tensor shape: {obs_tensor.shape}")
        
        # Move tensor to the same device as the model
        device = next(policy.parameters()).device
        obs_tensor = obs_tensor.to(device)
        print(f"Tensor device: {obs_tensor.device}, Model device: {device}")
        
        # Get policy's chosen action
        with torch.no_grad():
            q_values = policy(obs_tensor)
            chosen_action = q_values.argmax(dim=1).item()
            q_val_chosen = q_values[0, chosen_action].item()
        
        print(f"Chosen action: {chosen_action}")
        print(f"Q-value for chosen action: {q_val_chosen:.4f}")
        
        # Get attributions for all frames at once (much more efficient!)
        all_attributions, delta = get_all_frame_attributions(obs_tensor, policy, chosen_action)
        
        # Extract individual frame attributions
        frame_attributions = []
        for frame_idx in range(4):
            attr_frame = all_attributions[0, frame_idx].detach().cpu().numpy()  # (H, W)
            frame_attributions.append(attr_frame)
            
            # Print frame attribution stats
            print(f"Frame {frame_idx} attribution: mean={attr_frame.mean():.4f}, std={attr_frame.std():.4f}")
        
        print(f"Convergence delta: {delta.item():.6f}")
        
        # Get RGB frame for visualization
        rgb_frame = env.envs[0].render()
        
        # Visualize every 5 steps or at the end
        if step % SAVE_INTERVAL == 0 or step == video_length:
            # Standard visualization with RGB
            save_path_standard = f"{EXP_DIR}/attribution_step_{step}_action_{chosen_action}.png"
            visualize_frame_attribution(
                obs_tensor, 
                frame_attributions, 
                chosen_action, 
                step, 
                rgb_frame=rgb_frame,
                save_path=save_path_standard
            )
            
            # RGB overlay visualization
            save_path_overlay = f"{EXP_DIR}/rgb_overlay_step_{step}_action_{chosen_action}.png"
            visualize_rgb_attribution_overlay(
                obs_tensor,
                frame_attributions,
                chosen_action,
                step,
                rgb_frame,
                save_path=save_path_overlay
            )
            simple_test_plot(frame_attributions, obs_tensor, 4)
        
        # Take action and get next observation
        obs, _, _, _ = env.step([chosen_action])
    
    env.close()
    print("\nAnalysis complete!")


if __name__ == "__main__":
    simple_attribution_analysis()
