import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from captum.attr import IntegratedGradients, DeepLift, DeepLiftShap, InputXGradient, GuidedBackprop, GradientShap, Saliency
from captum.attr import visualization as viz

import highway_env  # noqa: F401

import torch
import numpy as np

NUM_STEPS = int(1e5)
EXP_DIR_NAME = "highway_cnn_captum_ig"

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
        "highway-fast-v0",
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
        "highway-fast-v0",
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


def forward_func(obs_tensor, action=None):
    q_values = policy(obs_tensor)  # shape [batch, n_actions]
    
    if action is None:
        action = q_values.argmax(dim=1)
    
    # If action is a tensor, gather Q-values for that action
    if isinstance(action, th.Tensor):
        return q_values.gather(1, action.unsqueeze(1)).squeeze()
    else:
        return q_values[:, action]


def forward_func_single_frame(obs_tensor, frame_idx, action=None):
    """
    Forward function that processes only a single frame from the stack
    """
    # Extract single frame: obs_tensor shape is [batch, channels, height, width]
    # For stacked frames: [batch, 4, height, width] -> [batch, 1, height, width]
    single_frame = obs_tensor[:, frame_idx:frame_idx+1, :, :]
    
    # Create a modified input with only this frame (zero out others)
    modified_obs = th.zeros_like(obs_tensor)
    modified_obs[:, frame_idx:frame_idx+1, :, :] = single_frame
    
    q_values = policy(modified_obs)
    
    if action is None:
        action = q_values.argmax(dim=1)
    
    if isinstance(action, th.Tensor):
        return q_values.gather(1, action.unsqueeze(1)).squeeze()
    else:
        return q_values[:, action]


def forward_func_frame_importance(obs_tensor, frame_weights, action=None):
    """
    Forward function that weights different frames differently
    frame_weights: tensor of shape [4] with weights for each frame
    """
    # Apply weights to each frame
    weighted_obs = obs_tensor * frame_weights.view(1, -1, 1, 1)
    
    q_values = policy(weighted_obs)
    
    if action is None:
        action = q_values.argmax(dim=1)
    
    if isinstance(action, th.Tensor):
        return q_values.gather(1, action.unsqueeze(1)).squeeze()
    else:
        return q_values[:, action]


def compare_baseline_strategies(obs_tensor, policy, action=None):
    """
    Compare different baseline strategies for Integrated Gradients
    """
    print("=" * 60)
    print("BASELINE STRATEGY COMPARISON")
    print("=" * 60)
    
    if action is None:
        with torch.no_grad():
            q_values = policy(obs_tensor)
            action = q_values.argmax(dim=1).item()
    
    def forward_func_baseline(input_tensor):
        return forward_func(input_tensor, action)
    
    ig = IntegratedGradients(forward_func_baseline)
    
    # Different baseline strategies
    baselines = {
        'Zero Baseline': torch.zeros_like(obs_tensor),
        'Mean Baseline': torch.mean(obs_tensor, dim=0, keepdim=True),
        'Random Baseline': torch.randn_like(obs_tensor) * 0.1,
        'Gaussian Noise': torch.randn_like(obs_tensor) * obs_tensor.std(),
        'Uniform Noise': torch.rand_like(obs_tensor) * 0.5,
    }
    
    results = {}
    
    for baseline_name, baseline in baselines.items():
        print(f"\n{baseline_name}:")
        print(f"  Baseline stats: mean={baseline.mean():.4f}, std={baseline.std():.4f}")
        
        # Compute attributions with this baseline
        attributions = ig.attribute(
            obs_tensor, 
            baselines=baseline, 
            target=0, 
            n_steps=50
        )
        
        # Analyze attribution statistics
        attr_stats = {
            'mean': attributions.mean().item(),
            'std': attributions.std().item(),
            'min': attributions.min().item(),
            'max': attributions.max().item(),
            'l2_norm': torch.norm(attributions).item()
        }
        
        results[baseline_name] = {
            'attributions': attributions,
            'stats': attr_stats
        }
        
        print(f"  Attribution stats:")
        for stat_name, stat_value in attr_stats.items():
            print(f"    {stat_name}: {stat_value:.4f}")
    
    # Compare results
    print(f"\n{'Baseline':<20} {'Mean':<10} {'Std':<10} {'L2 Norm':<10}")
    print("-" * 50)
    for baseline_name, result in results.items():
        stats = result['stats']
        print(f"{baseline_name:<20} {stats['mean']:<10.4f} {stats['std']:<10.4f} {stats['l2_norm']:<10.4f}")
    
    return results


def analyze_frame_attributions(obs_tensor, policy, action=None):
    """
    Analyze attributions for each individual frame in the stack
    """
    print("=" * 60)
    print("FRAME-WISE ATTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Get the predicted action
    with torch.no_grad():
        q_values = policy(obs_tensor)
        if action is None:
            action = q_values.argmax(dim=1).item()
        print(f"Analyzing action: {action}")
        print(f"Q-values: {q_values.squeeze()}")
    
    # Method 1: Individual frame analysis
    print("\n1. INDIVIDUAL FRAME ANALYSIS")
    print("-" * 40)
    
    frame_attributions = []
    for frame_idx in range(4):  # Assuming 4 stacked frames
        print(f"\nAnalyzing Frame {frame_idx} (most recent = 3):")
        
        # Create forward function for this specific frame
        def forward_single_frame(input_tensor):
            return forward_func_single_frame(input_tensor, frame_idx, action)
        
        # Compute attributions for this frame only
        ig = IntegratedGradients(forward_single_frame)
        
        # Different baseline options:
        # Option 1: Zero baseline (most common)
        zero_baseline = torch.zeros_like(obs_tensor)
        
        # Option 2: Mean baseline (average of all observations)
        # mean_baseline = torch.mean(obs_tensor, dim=0, keepdim=True)
        
        # Option 3: Random baseline
        # random_baseline = torch.randn_like(obs_tensor)
        
        # Option 4: Black image baseline (for visual data)
        # black_baseline = torch.zeros_like(obs_tensor)
        
        attributions = ig.attribute(
            obs_tensor, 
            baselines=zero_baseline,  # Specify the baseline
            target=0, 
            n_steps=50
        )
        
        # Get attribution for this specific frame
        frame_attr = attributions[0, frame_idx, :, :].detach().cpu().numpy()
        frame_attributions.append(frame_attr)
        
        print(f"  Frame {frame_idx} attribution stats:")
        print(f"    Mean: {frame_attr.mean():.4f}")
        print(f"    Std: {frame_attr.std():.4f}")
        print(f"    Min: {frame_attr.min():.4f}")
        print(f"    Max: {frame_attr.max():.4f}")
    
    # Method 2: Frame importance analysis
    print("\n2. FRAME IMPORTANCE ANALYSIS")
    print("-" * 40)
    
    # Create baseline (all frames zero)
    baseline = torch.zeros_like(obs_tensor)
    
    # Test each frame individually
    frame_importances = []
    for frame_idx in range(4):
        # Create input with only this frame
        test_input = baseline.clone()
        test_input[0, frame_idx, :, :] = obs_tensor[0, frame_idx, :, :]
        
        # Get Q-value for this frame only
        with torch.no_grad():
            q_val_single = policy(test_input)[0, action].item()
            q_val_baseline = policy(baseline)[0, action].item()
            importance = q_val_single - q_val_baseline
        
        frame_importances.append(importance)
        print(f"Frame {frame_idx} importance: {importance:.4f}")
    
    # Method 3: Gradient-based frame analysis
    print("\n3. GRADIENT-BASED FRAME ANALYSIS")
    print("-" * 40)
    
    obs_tensor.requires_grad_(True)
    
    def forward_with_grad(input_tensor):
        return forward_func(input_tensor, action)
    
    # Compute gradients
    q_val = forward_with_grad(obs_tensor)
    q_val.backward()
    
    # Get gradients for each frame
    gradients = obs_tensor.grad[0]  # Shape: [4, 128, 64]
    
    for frame_idx in range(4):
        frame_grad = gradients[frame_idx, :, :].detach().cpu().numpy()
        print(f"Frame {frame_idx} gradient stats:")
        print(f"    Mean: {frame_grad.mean():.4f}")
        print(f"    Std: {frame_grad.std():.4f}")
        print(f"    L2 norm: {np.linalg.norm(frame_grad):.4f}")
    
    return {
        'frame_attributions': frame_attributions,
        'frame_importances': frame_importances,
        'gradients': gradients.detach().cpu().numpy()
    }


def visualize_frame_attributions(obs_tensor, frame_attributions, save_path="frame_attributions"):
    """
    Visualize attributions for each frame
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for frame_idx in range(4):
        # Original frame
        original_frame = obs_tensor[0, frame_idx, :, :].detach().cpu().numpy()
        axes[0, frame_idx].imshow(original_frame, cmap='gray')
        axes[0, frame_idx].set_title(f'Original Frame {frame_idx}')
        axes[0, frame_idx].axis('off')
        
        # Attribution map
        attr_map = frame_attributions[frame_idx]
        im = axes[1, frame_idx].imshow(attr_map, cmap='RdBu_r', alpha=0.7)
        axes[1, frame_idx].set_title(f'Attribution Frame {frame_idx}')
        axes[1, frame_idx].axis('off')
        plt.colorbar(im, ax=axes[1, frame_idx])
    
    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=150, bbox_inches='tight')
    plt.show()


def captum_analysis_episode():
    """
    Run attribution analysis for the policy's chosen actions throughout an episode
    """
    model = DQN.load(f"{EXP_DIR_NAME}/model")
    policy = model.q_net

    env = DummyVecEnv([test_env])
    video_length = 2 * env.envs[0].unwrapped.config["duration"]
    env = VecVideoRecorder(
        env,
        f"{EXP_DIR_NAME}/videos/",
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix="dqn-agent-captum",
    )
    obs = env.reset()

    # Storage for episode data
    episode_data = {
        'observations': [],
        'actions': [],
        'q_values': [],
        'frame_attributions': [],
        'frame_importances': [],
        'rgb_frames': []
    }
    
    print("=" * 60)
    print("EPISODE ATTRIBUTION ANALYSIS")
    print("=" * 60)
    print(f"Running for {video_length + 1} steps...")

    for step in range(video_length + 1):
        print(f"\nStep {step}/{video_length}")
        print("-" * 30)
        
        # Convert observation to tensor
        obs_tensor = torch.tensor(obs).float().unsqueeze(0)
        
        # Get policy's chosen action and Q-values
        with torch.no_grad():
            q_values = policy(obs_tensor)
            chosen_action = q_values.argmax(dim=1).item()
            q_val_chosen = q_values[0, chosen_action].item()
        
        print(f"Chosen action: {chosen_action}")
        print(f"Q-values: {q_values.squeeze().numpy()}")
        print(f"Q-value for chosen action: {q_val_chosen:.4f}")
        
        # Get RGB frame for visualization
        rgb_frame = env.envs[0].render()
        
        # Perform attribution analysis for the chosen action
        step_results = analyze_frame_attributions(obs_tensor, policy, action=chosen_action)
        
        # Store episode data
        episode_data['observations'].append(obs_tensor.clone())
        episode_data['actions'].append(chosen_action)
        episode_data['q_values'].append(q_values.squeeze().numpy())
        episode_data['frame_attributions'].append(step_results['frame_attributions'])
        episode_data['frame_importances'].append(step_results['frame_importances'])
        episode_data['rgb_frames'].append(rgb_frame)
        
        # Take action and get next observation
        obs, _, _, _ = env.step([chosen_action])
        
        # Save attribution visualization for key steps
        if step % 5 == 0 or step in [0, video_length]:
            visualize_frame_attributions(
                obs_tensor, 
                step_results['frame_attributions'],
                save_path=f"attributions_step_{step}_action_{chosen_action}"
            )

    env.close()
    
    # Analyze episode results
    analyze_episode_results(episode_data)
    
    return episode_data


def analyze_episode_results(episode_data):
    """
    Analyze the attribution results across the entire episode
    """
    print("\n" + "=" * 60)
    print("EPISODE ANALYSIS SUMMARY")
    print("=" * 60)
    
    actions = episode_data['actions']
    frame_importances = np.array(episode_data['frame_importances'])  # Shape: [steps, 4]
    
    # Action distribution
    print("\n1. ACTION DISTRIBUTION:")
    unique_actions, counts = np.unique(actions, return_counts=True)
    for action, count in zip(unique_actions, counts):
        print(f"  Action {action}: {count} times ({count/len(actions)*100:.1f}%)")
    
    # Frame importance analysis
    print("\n2. FRAME IMPORTANCE ACROSS EPISODE:")
    print("   Frame 0 (oldest)  Frame 1  Frame 2  Frame 3 (newest)")
    print(f"   Mean: {frame_importances.mean(axis=0)}")
    print(f"   Std:  {frame_importances.std(axis=0)}")
    print(f"   Max:  {frame_importances.max(axis=0)}")
    
    # Temporal patterns
    print("\n3. TEMPORAL PATTERNS:")
    for frame_idx in range(4):
        frame_importance_series = frame_importances[:, frame_idx]
        print(f"Frame {frame_idx} importance over time:")
        print(f"  Trend: {'increasing' if np.corrcoef(range(len(frame_importance_series)), frame_importance_series)[0,1] > 0 else 'decreasing'}")
        print(f"  Variability: {frame_importance_series.std():.4f}")
    
    # Action-specific analysis
    print("\n4. ACTION-SPECIFIC FRAME IMPORTANCE:")
    for action in unique_actions:
        action_mask = np.array(actions) == action
        if np.sum(action_mask) > 1:  # Only if action appears multiple times
            action_frame_importance = frame_importances[action_mask]
            print(f"Action {action} (n={np.sum(action_mask)}):")
            print(f"  Frame importance: {action_frame_importance.mean(axis=0)}")
    
    # Create episode visualization
    create_episode_visualization(episode_data)


def create_episode_visualization(episode_data):
    """
    Create a comprehensive visualization of the episode
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Plot 1: Action sequence
    axes[0, 0].plot(episode_data['actions'], 'o-', markersize=4)
    axes[0, 0].set_title('Chosen Actions Over Time')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Action')
    axes[0, 0].grid(True)
    
    # Plot 2: Q-values for chosen actions
    chosen_q_values = [q_vals[action] for q_vals, action in zip(episode_data['q_values'], episode_data['actions'])]
    axes[0, 1].plot(chosen_q_values, 'o-', markersize=4)
    axes[0, 1].set_title('Q-values for Chosen Actions')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Q-value')
    axes[0, 1].grid(True)
    
    # Plot 3: Frame importance over time
    frame_importances = np.array(episode_data['frame_importances'])
    for frame_idx in range(4):
        axes[1, 0].plot(frame_importances[:, frame_idx], label=f'Frame {frame_idx}', alpha=0.7)
    axes[1, 0].set_title('Frame Importance Over Time')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Importance')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Frame importance heatmap
    im = axes[1, 1].imshow(frame_importances.T, aspect='auto', cmap='viridis')
    axes[1, 1].set_title('Frame Importance Heatmap')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Frame')
    axes[1, 1].set_yticks(range(4))
    axes[1, 1].set_yticklabels([f'Frame {i}' for i in range(4)])
    plt.colorbar(im, ax=axes[1, 1])
    
    # Plot 5: Action distribution
    unique_actions, counts = np.unique(episode_data['actions'], return_counts=True)
    axes[2, 0].bar(unique_actions, counts)
    axes[2, 0].set_title('Action Distribution')
    axes[2, 0].set_xlabel('Action')
    axes[2, 0].set_ylabel('Count')
    
    # Plot 6: Average frame importance by action
    frame_importances = np.array(episode_data['frame_importances'])
    actions = np.array(episode_data['actions'])
    unique_actions = np.unique(actions)
    
    action_frame_importance = []
    for action in unique_actions:
        mask = actions == action
        if np.sum(mask) > 0:
            avg_importance = frame_importances[mask].mean(axis=0)
            action_frame_importance.append(avg_importance)
        else:
            action_frame_importance.append(np.zeros(4))
    
    if len(action_frame_importance) > 0:
        action_frame_importance = np.array(action_frame_importance)
        im = axes[2, 1].imshow(action_frame_importance, aspect='auto', cmap='viridis')
        axes[2, 1].set_title('Average Frame Importance by Action')
        axes[2, 1].set_xlabel('Frame')
        axes[2, 1].set_ylabel('Action')
        axes[2, 1].set_xticks(range(4))
        axes[2, 1].set_xticklabels([f'Frame {i}' for i in range(4)])
        axes[2, 1].set_yticks(range(len(unique_actions)))
        axes[2, 1].set_yticklabels([f'Action {a}' for a in unique_actions])
        plt.colorbar(im, ax=axes[2, 1])
    
    plt.tight_layout()
    plt.savefig("episode_attribution_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()


def captum_analysis():
    """
    Main function to run attribution analysis for policy's chosen actions
    """
    return captum_analysis_episode()




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
    model.learn(total_timesteps=NUM_STEPS)
    model.save(f"{EXP_DIR_NAME}/model")
    
    # Analyze the trained model
    model_info = analyze_model(model)


if __name__ == "__main__":
    # Train
    # train()

    # Inference
    inference()

    # captum analysis
    # captum_analysis()
