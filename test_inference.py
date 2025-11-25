import torch
import numpy as np
import os
import pickle
import argparse
import sys
import time
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from einops import rearrange

from constants import I2RT_TASK_CONFIGS, JOINT_NAMES, DT
from utils import load_episode, set_seed

import IPython
e = IPython.embed

# Optional imports for simulation
try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("Warning: mujoco not available. Simulation will be disabled.")

try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False
    print("Warning: rerun not available. Visualization will be limited.")

def make_policy(policy_class, policy_config):
    """Create policy instance. Import here to avoid argument parsing issues."""
    from policy import ACTPolicy, CNNMLPPolicy
    print(policy_config)
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def load_model_and_stats(ckpt_dir, policy_class, policy_config):
    """Load pretrained model and dataset statistics."""
    ckpt_path = os.path.join(ckpt_dir, 'policy_epoch_1000_seed_0.ckpt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    # Temporarily replace sys.argv to avoid argument parsing issues in build_ACT_model_and_optimizer
    # The build functions call parser.parse_args() which reads from sys.argv
    original_argv = sys.argv
    try:
        # Set dummy arguments that satisfy the parser requirements
        dummy_args = [
            'test_inference.py',
            '--ckpt_dir', ckpt_dir,
            '--policy_class', policy_class,
            '--task_name', 'i2rt_cup_to_plate',
            '--seed', '0',
            '--num_epochs', '1',
        ]
        # Add optional arguments if they're in policy_config
        if 'kl_weight' in policy_config:
            dummy_args.extend(['--kl_weight', str(policy_config['kl_weight'])])
        if 'num_queries' in policy_config:
            dummy_args.extend(['--chunk_size', str(policy_config['num_queries'])])
        
        sys.argv = dummy_args
        policy = make_policy(policy_class, policy_config)
    finally:
        sys.argv = original_argv
    
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    print(f"Model loading status: {loading_status}")
    policy.cuda()
    policy.eval()
    print(f'Loaded model from: {ckpt_path}')
    
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found at {stats_path}")
    
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    print(f'Loaded stats from: {stats_path}')
    
    return policy, stats

def preprocess_observation(qpos, images, camera_names, stats):
    """Preprocess observation for model input.
    
    Images from load_episode are (H, W, C) uint8 [0, 255].
    Model expects (num_cameras, C, H, W) float [0, 1].
    Policy will apply ImageNet normalization internally.
    """
    # Normalize qpos
    qpos_normalized = (qpos - stats['qpos_mean']) / stats['qpos_std']
    qpos_tensor = torch.from_numpy(qpos_normalized).float().cuda().unsqueeze(0)
    
    # Process images to match dataset format: (num_cameras, C, H, W) with values in [0, 1]
    image_list = []
    for cam_name in camera_names:
        if cam_name in images:
            img = images[cam_name]
            # Image from load_episode is (H, W, C) uint8 [0, 255]
            # Convert to (H, W, C) float [0, 1]
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif img.dtype in [np.float32, np.float64]:
                # Already float, check if needs scaling
                if img.max() > 1.0:
                    img = img.astype(np.float32) / 255.0
                else:
                    img = img.astype(np.float32)
            
            # Ensure channel-last format (H, W, C)
            if img.ndim == 3 and img.shape[0] == 3:  # Already channel-first (C, H, W)
                img = np.transpose(img, (1, 2, 0))  # Convert to (H, W, C)
            
            image_list.append(img)
        else:
            raise KeyError(f"Camera '{cam_name}' not found in images. Available: {list(images.keys())}")
    
    # Stack: (num_cameras, H, W, C), then convert to (num_cameras, C, H, W)
    all_cam_images = np.stack(image_list, axis=0)  # (num_cameras, H, W, C)
    image_data = torch.from_numpy(all_cam_images).float()
    # Convert channel-last to channel-first: (num_cameras, H, W, C) -> (num_cameras, C, H, W)
    image_data = torch.einsum('k h w c -> k c h w', image_data)
    # Add batch dimension: (1, num_cameras, C, H, W)
    image_tensor = image_data.unsqueeze(0).cuda()
    
    return qpos_tensor, image_tensor

def run_inference_on_episode(policy, episode_data, camera_names, stats, query_frequency=1, temporal_agg=False, num_queries=100, state_dim=14):
    """Run inference on a single episode and return predicted actions.
    
    Args:
        policy: The policy model
        episode_data: Tuple of (qpos_seq, qvel_seq, action_seq_gt, image_dict)
        camera_names: List of camera names
        stats: Normalization statistics
        query_frequency: How often to query the policy (default: 1)
        temporal_agg: Whether to use temporal aggregation (default: False)
        num_queries: Number of queries in action chunk (for temporal agg)
        state_dim: State dimension (for temporal agg buffer)
    """
    qpos_seq, _, action_seq_gt, image_dict = episode_data
    
    predicted_actions = []
    ground_truth_actions = []
    
    num_timesteps = len(qpos_seq)
    
    # Setup temporal aggregation if enabled
    if temporal_agg:
        query_frequency = 1  # Query every timestep when using temporal agg
        all_time_actions = torch.zeros([num_timesteps, num_timesteps + num_queries, state_dim]).cuda()
    
    action_chunk = None
    chunk_start_idx = 0
    
    with torch.inference_mode():
        for t in tqdm(range(num_timesteps), desc="Running inference"):
            # Get current observation
            qpos = qpos_seq[t]
            images = {cam: image_dict[cam][t] for cam in camera_names if cam in image_dict}
            
            # Preprocess
            qpos_tensor, image_tensor = preprocess_observation(qpos, images, camera_names, stats)
            
            # Run inference when needed
            if t % query_frequency == 0:
                # Get action prediction (returns chunk of actions)
                action_chunk_tensor = policy(qpos_tensor, image_tensor)
                
                # For ACT, action_chunk is (batch, num_queries, action_dim)
                if action_chunk_tensor.ndim == 3:
                    action_chunk_tensor = action_chunk_tensor[0]  # (num_queries, action_dim)
                else:
                    action_chunk_tensor = action_chunk_tensor[0]
                    if action_chunk_tensor.ndim == 1:
                        action_chunk_tensor = action_chunk_tensor.reshape(1, -1)
                
                if temporal_agg:
                    # Store in temporal aggregation buffer
                    # Ensure t is an integer for indexing
                    t_int = int(t)
                    all_time_actions[[t_int], t_int:t_int+num_queries] = action_chunk_tensor
                else:
                    # Convert to numpy for regular inference
                    action_chunk = action_chunk_tensor.cpu().numpy()
                    chunk_start_idx = t
            
            # Get action based on temporal aggregation or regular mode
            if temporal_agg:
                # Collect all predictions for current timestep t
                # Ensure t is an integer for indexing
                t_int = int(t)
                actions_for_curr_step = all_time_actions[:, t_int]  # (num_timesteps, action_dim)
                
                # Filter out zero (unpopulated) predictions
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                
                if len(actions_for_curr_step) > 0:
                    # Apply exponential weighting (recent = higher weight)
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    
                    # Weighted average
                    action_pred_normalized = (actions_for_curr_step * exp_weights).sum(dim=0).cpu().numpy()
                else:
                    # Fallback if no predictions available
                    action_pred_normalized = np.zeros(state_dim)
            else:
                # Regular mode: get action from chunk
                chunk_idx = min(t - chunk_start_idx, action_chunk.shape[0] - 1)
                action_pred_normalized = action_chunk[chunk_idx]
            
            # Denormalize predicted action
            action_pred_denorm = action_pred_normalized * stats['action_std'] + stats['action_mean']
            predicted_actions.append(action_pred_denorm)
            
            # Store ground truth (raw actions from dataset, already denormalized)
            ground_truth_actions.append(action_seq_gt[t])
    
    return np.array(predicted_actions), np.array(ground_truth_actions)

def plot_action_comparison(predicted_actions, ground_truth_actions, output_path, joint_names=None):
    """Plot action comparison per joint dimension."""
    num_joints = predicted_actions.shape[1]
    num_timesteps = predicted_actions.shape[0]
    
    # Create joint names if not provided
    if joint_names is None:
        joint_names = [f'Joint {i}' for i in range(num_joints)]
    
    # Determine layout: 2 columns, enough rows
    n_cols = 2
    n_rows = (num_joints + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    timesteps = np.arange(num_timesteps)
    
    for joint_idx in range(num_joints):
        ax = axes[joint_idx]
        
        # Plot ground truth and predicted
        ax.plot(timesteps, ground_truth_actions[:, joint_idx], 
                label='Ground Truth', linewidth=2, alpha=0.7)
        ax.plot(timesteps, predicted_actions[:, joint_idx], 
                label='Predicted', linewidth=2, alpha=0.7, linestyle='--')
        
        ax.set_title(f'{joint_names[joint_idx]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Timestep', fontsize=10)
        ax.set_ylabel('Action Value', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_joints, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved action comparison plot to: {output_path}')
    plt.close()

def plot_action_error(predicted_actions, ground_truth_actions, output_path, joint_names=None):
    """Plot absolute error per joint over time."""
    num_joints = predicted_actions.shape[1]
    num_timesteps = predicted_actions.shape[0]
    
    if joint_names is None:
        joint_names = [f'Joint {i}' for i in range(num_joints)]
    
    errors = np.abs(predicted_actions - ground_truth_actions)
    
    n_cols = 2
    n_rows = (num_joints + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    timesteps = np.arange(num_timesteps)
    
    for joint_idx in range(num_joints):
        ax = axes[joint_idx]
        
        ax.plot(timesteps, errors[:, joint_idx], 
                label='Absolute Error', linewidth=2, color='red', alpha=0.7)
        
        # Add mean error line
        mean_error = np.mean(errors[:, joint_idx])
        ax.axhline(y=mean_error, color='orange', linestyle='--', 
                  label=f'Mean Error: {mean_error:.4f}', linewidth=2)
        
        ax.set_title(f'{joint_names[joint_idx]} - Error', fontsize=12, fontweight='bold')
        ax.set_xlabel('Timestep', fontsize=10)
        ax.set_ylabel('Absolute Error', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    for idx in range(num_joints, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved error plot to: {output_path}')
    plt.close()

def load_mujoco_model(mjcf_path):
    """Load MuJoCo model, data, and renderer."""
    if not MUJOCO_AVAILABLE:
        raise ImportError("MuJoCo is not available. Install with: pip install mujoco")
    
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    return model, data, renderer

def to_hwc_uint8_numpy(tensor):
    """Convert CHW float32 [0,1] tensor to HWC uint8 numpy array."""
    if isinstance(tensor, np.ndarray):
        if tensor.ndim == 3 and tensor.shape[0] == 3:  # C, H, W
            tensor = np.transpose(tensor, (1, 2, 0))  # H, W, C
        if tensor.dtype != np.uint8:
            tensor = (np.clip(tensor, 0, 1) * 255).astype(np.uint8)
        return tensor
    else:
        # torch.Tensor
        if tensor.ndim == 3 and tensor.shape[0] == 3:  # C, H, W
            tensor = tensor.permute(1, 2, 0)  # H, W, C
        return (tensor.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

def log_actions_to_rerun(u_pred, u_gt, timestep):
    """Log action comparison to Rerun."""
    if not RERUN_AVAILABLE:
        return
    
    u_pred = np.asarray(u_pred, dtype=np.float32).reshape(-1)
    u_gt = np.asarray(u_gt, dtype=np.float32).reshape(-1)
    n = min(len(u_pred), len(u_gt))
    
    for i in range(n):
        rr.log(f"actions/gt/joint_{i}", rr.Scalars([float(u_gt[i])]))
        rr.log(f"actions/pred/joint_{i}", rr.Scalars([float(u_pred[i])]))

def run_inference_with_simulation(
    policy, episode_data, camera_names, stats, 
    mjcf_path, dataset_dir, episode_idx, fps, query_frequency=1, use_rerun=False, use_predicted_action=True,
    temporal_agg=False, num_queries=100, state_dim=14
):
    """Run inference on episode with MuJoCo simulation in real-time.
    
    This function loads the dataset directly to get proper timestamps and actions,
    matching the working example format.
    """
    if not MUJOCO_AVAILABLE:
        raise ImportError("MuJoCo is not available for simulation")
    
    # Load dataset directly to get proper timestamps and actions (like working example)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils.constants import ACTION
    except ImportError:
        raise ImportError("LeRobot package required for simulation. Install with: pip install lerobot")
    
    # Resolve dataset path
    from utils import _resolve_lerobot_repo_and_root
    repo_id, root = _resolve_lerobot_repo_and_root(dataset_dir)
    ds = LeRobotDataset(repo_id=repo_id, root=root, episodes=[episode_idx])
    
    # Ensure indices are integers
    # Access episode metadata correctly (matching utils.py pattern)
    episode_meta = ds.meta.episodes[episode_idx]
    # Ensure indices are integers (convert from any numeric type)
    from_idx = int(float(episode_meta["dataset_from_index"]))
    to_idx = int(float(episode_meta["dataset_to_index"]))
    
    # Load MuJoCo model
    model, data, renderer = load_mujoco_model(mjcf_path)
    sim_dt = model.opt.timestep
    
    # Initialize Rerun if requested
    if use_rerun and RERUN_AVAILABLE:
        rr.init(f"test_inference/episode_{episode_idx}", spawn=True)
    
    predicted_actions = []
    ground_truth_actions = []
    
    num_timesteps = to_idx - from_idx
    
    # Setup temporal aggregation if enabled
    if temporal_agg:
        query_frequency = 1  # Query every timestep when using temporal agg
        all_time_actions = torch.zeros([num_timesteps, num_timesteps + num_queries, state_dim]).cuda()
    
    action_chunk = None
    chunk_start_idx = 0
    first_ts = None
    t0_wall = time.time()
    
    print(f"Running inference with MuJoCo simulation ({num_timesteps} timesteps)...")
    if temporal_agg:
        print("Temporal aggregation enabled: querying policy every timestep")
    
    with torch.inference_mode():
        for k in tqdm(range(from_idx, to_idx), desc="Simulation"):
            # Ensure k is an integer for dataset access
            k = int(k)
            sample = ds[k]
            ts = float(sample["timestamp"].item())
            frame_idx = int(k - from_idx)  # Ensure frame_idx is an integer
            
            # Timing for real-time playback
            if first_ts is None:
                first_ts = ts
                t0_wall = time.time()
            
            # Set Rerun time
            if use_rerun and RERUN_AVAILABLE:
                rr.set_time_seconds("timestamp", ts)
            
            # Log dataset images to Rerun
            if use_rerun and RERUN_AVAILABLE:
                for cam_key in ds.meta.camera_keys:
                    if cam_key in sample:
                        img_tensor = sample[cam_key]
                        if isinstance(img_tensor, torch.Tensor):
                            img_rr = to_hwc_uint8_numpy(img_tensor)
                        else:
                            img_rr = to_hwc_uint8_numpy(torch.from_numpy(np.array(img_tensor)))
                        rr.log(f"dataset/{cam_key}", rr.Image(img_rr))
            
            # Get ground truth action (raw from dataset, same format as working example)
            u_gt = sample[ACTION].numpy() if isinstance(sample[ACTION], torch.Tensor) else np.array(sample[ACTION])
            ground_truth_actions.append(u_gt.copy())
            
            # Get observation for inference
            # Extract qpos and images from sample
            qpos = sample.get('observation.state', None)
            if qpos is None:
                # Try alternative key
                qpos = sample.get('qpos', None)
            if qpos is None:
                raise KeyError("Could not find qpos/state in sample")
            
            if isinstance(qpos, torch.Tensor):
                qpos = qpos.cpu().numpy()
            else:
                qpos = np.array(qpos)
            
            images = {}
            for cam_name in camera_names:
                # Try different possible keys
                for key in [cam_name, f"observation.images.{cam_name}"]:
                    if key in sample:
                        img = sample[key]
                        if isinstance(img, torch.Tensor):
                            img = img.cpu().numpy()
                        else:
                            img = np.array(img)
                        # Convert to HWC uint8 format
                        if img.ndim == 3 and img.shape[0] == 3:  # C, H, W
                            img = np.transpose(img, (1, 2, 0))  # H, W, C
                        if img.dtype != np.uint8:
                            if img.max() <= 1.0:
                                img = (img * 255).astype(np.uint8)
                            else:
                                img = img.astype(np.uint8)
                        images[cam_name] = img
                        break
            
            # Preprocess for inference
            qpos_tensor, image_tensor = preprocess_observation(qpos, images, camera_names, stats)
            
            # Run inference when needed
            # Ensure frame_idx is an integer
            frame_idx = int(frame_idx)
            if frame_idx % query_frequency == 0:
                action_chunk_tensor = policy(qpos_tensor, image_tensor)
                
                # For ACT, action_chunk is (batch, num_queries, action_dim)
                if action_chunk_tensor.ndim == 3:
                    action_chunk_tensor = action_chunk_tensor[0]  # (num_queries, action_dim)
                else:
                    action_chunk_tensor = action_chunk_tensor[0]
                    if action_chunk_tensor.ndim == 1:
                        action_chunk_tensor = action_chunk_tensor.reshape(1, -1)
                
                if temporal_agg:
                    # Store in temporal aggregation buffer
                    # Ensure frame_idx is an integer for indexing
                    frame_idx_int = int(frame_idx)
                    all_time_actions[[frame_idx_int], frame_idx_int:frame_idx_int+num_queries] = action_chunk_tensor
                else:
                    # Convert to numpy for regular inference
                    action_chunk = action_chunk_tensor.cpu().numpy()
                    chunk_start_idx = frame_idx
            
            # Get action based on temporal aggregation or regular mode
            if temporal_agg:
                # Collect all predictions for current timestep
                # Ensure frame_idx is an integer for indexing
                frame_idx_int = int(frame_idx)
                actions_for_curr_step = all_time_actions[:, frame_idx_int]  # (num_timesteps, action_dim)
                
                # Filter out zero (unpopulated) predictions
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                
                if len(actions_for_curr_step) > 0:
                    # Apply exponential weighting (recent = higher weight)
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    
                    # Weighted average
                    action_pred_normalized = (actions_for_curr_step * exp_weights).sum(dim=0).cpu().numpy()
                else:
                    # Fallback if no predictions available
                    action_pred_normalized = np.zeros(state_dim)
            else:
                # Regular mode: get action from chunk
                chunk_idx = min(frame_idx - chunk_start_idx, action_chunk.shape[0] - 1)
                action_pred_normalized = action_chunk[chunk_idx]
            
            # Denormalize predicted action
            action_pred_denorm = action_pred_normalized * stats['action_std'] + stats['action_mean']
            predicted_actions.append(action_pred_denorm.copy())
            
            # Log actions to Rerun
            if use_rerun and RERUN_AVAILABLE:
                log_actions_to_rerun(action_pred_denorm, u_gt, frame_idx)
            
            # Calculate number of simulation steps until next frame (like working example)
            if k + 1 < to_idx:
                # Ensure k + 1 is an integer for dataset access
                k_next = int(k + 1)
                sample_next = ds[k_next]
                ts_next = float(sample_next["timestamp"].item())
            else:
                ts_next = ts + 1.0 / max(1.0, fps)
            n_steps = max(1, int(round((ts_next - ts) / sim_dt)))
            
            # Apply action to simulation (same format as working example)
            # Reorder: [left(7), right(7)] -> [right(7), left(7)]
            # Use predicted action or ground truth based on flag
            if use_predicted_action and len(action_pred_denorm) == 14:
                u_apply = np.concatenate([action_pred_denorm[7:], action_pred_denorm[:7]])
            else:
                # Use ground truth (same as working example) - useful for debugging
                u_apply = np.concatenate([u_gt[7:], u_gt[:7]])
            
            data.ctrl[:] = u_apply
            
            # Step simulation multiple times (like working example)
            for _ in range(n_steps):
                mujoco.mj_step(model, data)
            
            # Render and log simulation frame
            if use_rerun and RERUN_AVAILABLE:
                renderer.update_scene(data, camera=0)
                sim_image = renderer.render()
                rr.log("sim", rr.Image(sim_image))
    
    return np.array(predicted_actions), np.array(ground_truth_actions)

def main(args):
    set_seed(42)
    
    task_name = 'i2rt_cup_to_plate'
    task_config = I2RT_TASK_CONFIGS[task_name]
    
    if args.get('dataset_dir', None) is not None:
        dataset_dir = args.get('dataset_dir')
    else:
        dataset_dir = task_config['dataset_dir']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    num_episodes = task_config['num_episodes']
    
    # Model parameters (should match training config)
    policy_class = args.get('policy_class', 'ACT')
    ckpt_dir = args['ckpt_dir']
    episode_idx = args.get('episode_idx', 0)
    temporal_agg = args.get('temporal_agg', False)
    
    # Set query frequency: if temporal_agg, it will be set to 1 later
    # Otherwise, default to chunk_size (num_queries) like in imitate_episodes.py
    if not temporal_agg:
        query_frequency = args.get('query_frequency', args.get('chunk_size', 100))
    else:
        query_frequency = 1  # Will be set to 1 when temporal_agg is enabled
    
    # Fixed parameters (should match training)
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'
    
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            'lr': 1e-5,
            'num_queries': args.get('chunk_size', 100),
            'kl_weight': args.get('kl_weight', 10),
            'hidden_dim': args.get('hidden_dim', 512),
            'dim_feedforward': args.get('dim_feedforward', 3200),
            'lr_backbone': lr_backbone,
            'backbone': backbone,
            'enc_layers': enc_layers,
            'dec_layers': dec_layers,
            'nheads': nheads,
            'camera_names': camera_names,
        }
    elif policy_class == 'CNNMLP':
        policy_config = {
            'lr': 1e-5,
            'lr_backbone': lr_backbone,
            'backbone': backbone,
            'num_queries': 1,
            'camera_names': camera_names,
        }
    else:
        raise NotImplementedError(f"Policy class {policy_class} not supported")
    
    # Load model and stats
    print(f"\n{'='*60}")
    print(f"Loading model from: {ckpt_dir}")
    print(f"{'='*60}\n")
    policy, stats = load_model_and_stats(ckpt_dir, policy_class, policy_config)
    
    # Load episode data
    print(f"\n{'='*60}")
    print(f"Loading episode {episode_idx} from dataset")
    print(f"Dataset dir: {dataset_dir}")
    print(f"{'='*60}\n")
    
    try:
        episode_data = load_episode(
            dataset_dir,
            episode_idx,
            qpos_key='observation.state',
            action_key='action',
            camera_names=camera_names
        )
    except Exception as e:
        print(f"Error loading episode: {e}")
        print(f"\nTrying with default camera names...")
        episode_data = load_episode(
            dataset_dir,
            episode_idx,
            qpos_key='observation.state',
            action_key='action',
            camera_names=None
        )
        # Update camera_names to match what was actually loaded
        qpos, qvel, action, image_dict = episode_data
        camera_names = list(image_dict.keys())
        print(f"Using cameras: {camera_names}")
    
    qpos_seq, qvel_seq, action_seq_gt, image_dict = episode_data
    print(f"Episode loaded: {len(qpos_seq)} timesteps")
    print(f"QPOS shape: {qpos_seq.shape}")
    print(f"Action shape: {action_seq_gt.shape}")
    print(f"Cameras: {list(image_dict.keys())}")
    
    # Check if simulation is requested
    use_sim = args.get('sim', False)
    mjcf_path = args.get('mjcf_path', None)
    use_rerun = args.get('use_rerun', False)
    
    # Get temporal aggregation parameters
    num_queries = policy_config.get('num_queries', 100) if policy_class == 'ACT' else 1
    state_dim = 14  # Fixed for bimanual robot
    
    # Adjust query frequency based on temporal aggregation
    if temporal_agg:
        query_frequency = 1  # Query every timestep when using temporal agg
        print(f"Temporal aggregation enabled: will query policy every timestep")
    
    # Run inference
    print(f"\n{'='*60}")
    if use_sim:
        print(f"Running inference with MuJoCo simulation on episode {episode_idx}")
        if not MUJOCO_AVAILABLE:
            print("Warning: MuJoCo not available. Falling back to regular inference.")
            use_sim = False
        elif mjcf_path is None or not Path(mjcf_path).exists():
            print(f"Warning: MuJoCo XML file not found at {mjcf_path}. Falling back to regular inference.")
            use_sim = False
    else:
        print(f"Running inference on episode {episode_idx}")
    print(f"{'='*60}\n")
    
    if use_sim:
        # Get FPS from dataset if available
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from utils import _resolve_lerobot_repo_and_root
            repo_id, root = _resolve_lerobot_repo_and_root(dataset_dir)
            ds_temp = LeRobotDataset(repo_id=repo_id, root=root, episodes=[episode_idx])
            fps = float(ds_temp.meta.fps)
        except:
            fps = 1.0 / DT  # Fallback to DT-based FPS
            print(f"Warning: Could not get FPS from dataset, using {fps} Hz")
        
        use_predicted_action = not args.get('use_gt_action', False)  # Invert: --use_gt_action means use GT
        predicted_actions, ground_truth_actions = run_inference_with_simulation(
            policy, episode_data, camera_names, stats,
            mjcf_path=mjcf_path,
            dataset_dir=dataset_dir,
            episode_idx=episode_idx,
            fps=fps,
            query_frequency=query_frequency,
            use_rerun=use_rerun,
            use_predicted_action=use_predicted_action,
            temporal_agg=temporal_agg,
            num_queries=num_queries,
            state_dim=state_dim
        )
    else:
        predicted_actions, ground_truth_actions = run_inference_on_episode(
            policy, episode_data, camera_names, stats, 
            query_frequency=query_frequency,
            temporal_agg=temporal_agg,
            num_queries=num_queries,
            state_dim=state_dim
        )
    
    print(f"\nPredicted actions shape: {predicted_actions.shape}")
    print(f"Ground truth actions shape: {ground_truth_actions.shape}")
    
    # Compute statistics
    mse = np.mean((predicted_actions - ground_truth_actions) ** 2)
    mae = np.mean(np.abs(predicted_actions - ground_truth_actions))
    print(f"\nMean Squared Error: {mse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    
    # Create output directory
    output_dir = Path(ckpt_dir) / 'inference_results'
    output_dir.mkdir(exist_ok=True)
    
    # Generate joint names for bimanual robot (14 joints = 7 per arm)
    joint_names = []
    for side in ['left', 'right']:
        for joint_name in JOINT_NAMES + ['gripper']:
            joint_names.append(f'{joint_name}_{side}')
    
    # Plot action comparison
    comparison_path = output_dir / f'episode_{episode_idx}_action_comparison.png'
    plot_action_comparison(
        predicted_actions, 
        ground_truth_actions, 
        str(comparison_path),
        joint_names=joint_names[:predicted_actions.shape[1]]
    )
    
    # Plot error
    error_path = output_dir / f'episode_{episode_idx}_action_error.png'
    plot_action_error(
        predicted_actions,
        ground_truth_actions,
        str(error_path),
        joint_names=joint_names[:predicted_actions.shape[1]]
    )
    
    print(f"\n{'='*60}")
    print(f"Inference complete! Results saved to: {output_dir}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test inference on i2rt_cup_to_plate task')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Directory containing model checkpoint and stats')
    parser.add_argument('--episode_idx', type=int, default=0, help='Episode index to test (default: 0)')
    parser.add_argument('--policy_class', type=str, default='ACT', choices=['ACT', 'CNNMLP'], help='Policy class')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size for ACT (default: 100)')
    parser.add_argument('--kl_weight', type=float, default=10, help='KL weight for ACT (default: 10)')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension for ACT (default: 512)')
    parser.add_argument('--dim_feedforward', type=int, default=3200, help='Feedforward dimension for ACT (default: 3200)')
    parser.add_argument('--query_frequency', type=int, default=1, help='Query frequency for inference (default: 1, or num_queries if not using temporal_agg)')
    
    # Simulation options
    parser.add_argument('--sim', action='store_true', help='Enable MuJoCo simulation')
    parser.add_argument('--mjcf_path', type=str, default=None, help='Path to MuJoCo XML file (required if --sim)')
    parser.add_argument('--dataset_dir', type=str, default=None, help='Path to dataset directory (required if --sim)')
    parser.add_argument('--use_rerun', action='store_true', help='Use Rerun for visualization (requires rerun package)')
    parser.add_argument('--use_gt_action', action='store_true', help='Use ground truth actions instead of predictions (for debugging)')
    parser.add_argument('--temporal_agg', action='store_true', help='Enable temporal aggregation for smoother actions (automatically sets query_frequency=1, queries policy every timestep). Best option when computational limits are not a concern.')
    
    args = vars(parser.parse_args())
    
    # Validate simulation arguments
    if args.get('sim', False):
        if args.get('mjcf_path') is None:
            parser.error("--mjcf_path is required when --sim is enabled")
        if not Path(args['mjcf_path']).exists():
            parser.error(f"MuJoCo XML file not found: {args['mjcf_path']}")
    
    main(args)

