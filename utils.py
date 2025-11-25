import numpy as np
import torch
import os
import h5py
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def _looks_like_lerobot_dataset(dataset_dir):
    dataset_path = Path(dataset_dir)
    return dataset_path.is_dir() and (dataset_path / 'meta' / 'info.json').exists()


def _looks_like_hdf5_dataset(dataset_dir):
    dataset_path = Path(dataset_dir)
    if not dataset_path.is_dir():
        return False
    return any(dataset_path.glob('episode_*.hdf5'))


def _resolve_lerobot_repo_and_root(dataset_dir):
    dataset_path = Path(dataset_dir)
    if dataset_path.exists() and (dataset_path / 'meta' / 'info.json').exists():
        # Local dataset
        return None, str(dataset_path)
    else:
        # Remote dataset (repo_id)
        return str(dataset_dir), None


def load_episode(dataset_dir, episode_idx, qpos_key='observation.state', action_key='action', camera_names=None):
    """
    Load a single episode from either HDF5 or LeRobot dataset format.
    
    Args:
        dataset_dir: Path to dataset directory or LeRobot repo_id
        episode_idx: Episode index to load
        qpos_key: Key for qpos/state in LeRobot dataset (default: 'observation.state')
        action_key: Key for action in LeRobot dataset (default: 'action')
        camera_names: List of camera names to load (None = all cameras)
    
    Returns:
        tuple: (qpos, qvel, action, image_dict)
            - qpos: numpy array of shape (T, state_dim)
            - qvel: numpy array of shape (T, state_dim) or None for LeRobot datasets
            - action: numpy array of shape (T, action_dim)
            - image_dict: dict mapping camera names to numpy arrays of shape (T, H, W, C)
    """
    # Detect dataset format
    if _looks_like_lerobot_dataset(dataset_dir) or (not Path(dataset_dir).exists() and not _looks_like_hdf5_dataset(dataset_dir)):
        # LeRobot dataset
        return _load_episode_lerobot(dataset_dir, episode_idx, qpos_key, action_key, camera_names)
    else:
        # HDF5 dataset
        return _load_episode_hdf5(dataset_dir, episode_idx)


def _load_episode_hdf5(dataset_dir, episode_idx):
    """Load a single episode from HDF5 format."""
    dataset_name = f'episode_{episode_idx}'
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f'Dataset does not exist at {dataset_path}')

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        action = root['/action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    return qpos, qvel, action, image_dict


def _load_episode_lerobot(dataset_dir, episode_idx, qpos_key='observation.state', action_key='action', camera_names=None):
    """Load a single episode from LeRobot dataset format."""
    repo_id, root = _resolve_lerobot_repo_and_root(dataset_dir)
    dataset = LeRobotDataset(repo_id=repo_id, root=root, episodes=[episode_idx])
    
    # Get episode metadata to find frame range
    episode_meta = dataset.meta.episodes[episode_idx]
    start_idx = int(episode_meta["dataset_from_index"])
    end_idx = int(episode_meta["dataset_to_index"])
    
    # Resolve camera keys
    available_keys = dataset.meta.camera_keys
    if camera_names is None:
        camera_keys = list(available_keys)
    else:
        camera_keys = []
        for name in camera_names:
            if name in available_keys:
                camera_keys.append(name)
            elif f"observation.images.{name}" in available_keys:
                camera_keys.append(f"observation.images.{name}")
            else:
                raise ValueError(f"Camera key '{name}' not found. Available keys: {list(available_keys)}")
    
    # Extract all frames for this episode
    qpos_list = []
    action_list = []
    image_dict = {}
    
    # Initialize image dict with normalized camera names (remove observation.images. prefix for cleaner output)
    for cam_key in camera_keys:
        # Use simple name for output (e.g., "top" instead of "observation.images.top")
        simple_name = cam_key.replace("observation.images.", "") if cam_key.startswith("observation.images.") else cam_key
        image_dict[simple_name] = []
    
    for frame_idx in range(start_idx, end_idx):
        frame = dataset[frame_idx]
        
        # Extract qpos
        if qpos_key in frame:
            qpos_val = frame[qpos_key]
            if hasattr(qpos_val, 'cpu'):
                qpos_val = qpos_val.cpu().numpy()
            elif hasattr(qpos_val, 'numpy'):
                qpos_val = qpos_val.numpy()
            qpos_list.append(qpos_val)
        else:
            raise KeyError(f"Key '{qpos_key}' not found in dataset. Available keys: {list(frame.keys())}")
        
        # Extract action
        if action_key in frame:
            action_val = frame[action_key]
            if hasattr(action_val, 'cpu'):
                action_val = action_val.cpu().numpy()
            elif hasattr(action_val, 'numpy'):
                action_val = action_val.numpy()
            action_list.append(action_val)
        else:
            raise KeyError(f"Key '{action_key}' not found in dataset. Available keys: {list(frame.keys())}")
        
        # Extract images
        for cam_key in camera_keys:
            simple_name = cam_key.replace("observation.images.", "") if cam_key.startswith("observation.images.") else cam_key
            if cam_key in frame:
                img = frame[cam_key]
                # Convert torch tensor to numpy array
                if hasattr(img, 'cpu'):
                    img = img.cpu().numpy()
                elif hasattr(img, 'numpy'):
                    img = img.numpy()
                
                # Handle channel-first vs channel-last
                if img.ndim == 3 and img.shape[0] == 3:  # C, H, W
                    img = np.transpose(img, (1, 2, 0))  # H, W, C
                
                # Convert from [0, 1] float to [0, 255] uint8 if needed
                if img.dtype == np.float32 or img.dtype == np.float64:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                
                image_dict[simple_name].append(img)
            else:
                raise KeyError(f"Camera key '{cam_key}' not found in dataset frame.")
    
    # Convert lists to numpy arrays
    qpos = np.array(qpos_list)
    action = np.array(action_list)
    for simple_name in image_dict.keys():
        image_dict[simple_name] = np.array(image_dict[simple_name])
    
    # qvel not available in LeRobot format, return None
    qvel = None
    
    return qpos, qvel, action, image_dict


def _load_data_hdf5(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


def load_data(
    dataset_dir,
    num_episodes,
    camera_names,
    batch_size_train,
    batch_size_val,
    *,
    episode_len=None,
    is_sim_override=None,
    lerobot_kwargs=None,
):
    dataset_path = Path(dataset_dir)
    if _looks_like_lerobot_dataset(dataset_dir) or (not dataset_path.exists() and not _looks_like_hdf5_dataset(dataset_dir)):
        if episode_len is None:
            raise ValueError("episode_len must be provided when using a LeRobot dataset.")
        repo_id, root = _resolve_lerobot_repo_and_root(dataset_dir)
        lerobot_kwargs = lerobot_kwargs or {}
        return load_data_lerobot(
            repo_id=repo_id,
            episode_len=episode_len,
            batch_size_train=batch_size_train,
            batch_size_val=batch_size_val,
            camera_names=camera_names,
            root=root,
            num_episodes=num_episodes,
            is_sim=is_sim_override if is_sim_override is not None else False,
            lerobot_kwargs=lerobot_kwargs,
        )
    return _load_data_hdf5(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)


class LeRobotEpisodicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        lerobot_dataset,
        episode_ids,
        camera_names,
        norm_stats,
        episode_len,
        qpos_key,
        action_key,
    ):
        super(LeRobotEpisodicDataset).__init__()
        self.dataset = lerobot_dataset
        self.dataset._ensure_hf_dataset_loaded()
        self.episode_ids = [int(ep) for ep in episode_ids]
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.qpos_key = qpos_key
        self.action_key = action_key

        # convert stats to torch tensors for faster use
        self.action_mean = torch.from_numpy(norm_stats["action_mean"]).float()
        self.action_std = torch.from_numpy(norm_stats["action_std"]).float()
        self.qpos_mean = torch.from_numpy(norm_stats["qpos_mean"]).float()
        self.qpos_std = torch.from_numpy(norm_stats["qpos_std"]).float()

        # ensure correct shapes for broadcasting
        if self.action_mean.ndim == 1:
            self.action_mean = self.action_mean.unsqueeze(0)
        if self.action_std.ndim == 1:
            self.action_std = self.action_std.unsqueeze(0)
        self.episode_ranges = self._build_episode_ranges()
        if len(self.episode_ranges) == 0:
            raise ValueError("No valid episodes found in LeRobot dataset selection.")

        # build a lightweight dataset that only keeps action column for fast slicing
        action_columns_to_remove = [
            col for col in self.dataset.hf_dataset.column_names if col != self.action_key
        ]
        if len(action_columns_to_remove) == 0:
            self.action_only_dataset = self.dataset.hf_dataset
        else:
            self.action_only_dataset = self.dataset.hf_dataset.remove_columns(action_columns_to_remove)
        self.action_only_dataset.set_transform(None)
        self.action_only_dataset = self.action_only_dataset.with_format("torch")

    def _build_episode_ranges(self):
        episode_ranges = []
        for episode_id in self.episode_ids:
            episode_meta = self.dataset.meta.episodes[episode_id]
            start = int(episode_meta["dataset_from_index"])
            end = int(episode_meta["dataset_to_index"])
            if end - start <= 0:
                continue
            episode_ranges.append({"episode_index": episode_id, "start": start, "end": end})
        return episode_ranges

    def __len__(self):
        return len(self.episode_ranges)

    def _get_action_sequence(self, start_idx, stop_idx):
        actions = []
        for idx in range(start_idx, stop_idx):
            sample = self.action_only_dataset[idx]
            actions.append(sample[self.action_key])
        return torch.stack(actions, dim=0).float()

    def __getitem__(self, index):
        episode_info = self.episode_ranges[index]
        episode_start = episode_info["start"]
        episode_end = episode_info["end"]
        episode_length = episode_end - episode_start

        start_offset = np.random.randint(episode_length)
        frame_idx = episode_start + start_offset

        frame = self.dataset[frame_idx]
        qpos = frame[self.qpos_key].float()

        image_tensors = []
        for cam in self.camera_names:
            if cam not in frame:
                raise KeyError(f"Camera key '{cam}' not found in dataset sample.")
            image_tensors.append(frame[cam].float())
        if len(image_tensors) == 0:
            raise ValueError("No camera data available for the requested keys.")
        image_data = torch.stack(image_tensors, dim=0)

        available_future = episode_end - frame_idx
        action_horizon = min(self.episode_len, available_future)
        stop_idx = frame_idx + action_horizon
        action_seq = self._get_action_sequence(frame_idx, stop_idx)

        padded_action = torch.zeros((self.episode_len, action_seq.shape[-1]), dtype=torch.float32)
        padded_action[:action_seq.shape[0]] = action_seq

        is_pad = torch.ones(self.episode_len, dtype=torch.bool)
        is_pad[:action_seq.shape[0]] = False

        action_data = (padded_action - self.action_mean) / self.action_std
        qpos_data = (qpos - self.qpos_mean) / self.qpos_std

        return image_data, qpos_data, action_data, is_pad


def _resolve_camera_keys(requested_names, available_keys):
    if requested_names is None:
        if len(available_keys) == 0:
            raise ValueError("The LeRobot dataset does not expose any camera keys.")
        return list(available_keys)
    resolved = []
    for name in requested_names:
        if name in available_keys:
            resolved.append(name)
            continue
        prefixed = f"observation.images.{name}"
        if prefixed in available_keys:
            resolved.append(prefixed)
            continue
        raise ValueError(f"Camera key '{name}' not found. Available keys: {list(available_keys)}")
    return resolved


def _get_lerobot_norm_stats(lerobot_dataset, qpos_key, action_key):
    stats = lerobot_dataset.meta.stats
    if stats is None:
        raise ValueError("LeRobot dataset does not contain precomputed statistics.")

    def _extract(feature_key):
        if feature_key not in stats:
            raise ValueError(f"Statistics for '{feature_key}' not found in dataset metadata.")
        feature_stats = stats[feature_key]
        if "mean" not in feature_stats or "std" not in feature_stats:
            raise ValueError(f"'mean' and 'std' required for feature '{feature_key}'.")
        mean = np.array(feature_stats["mean"], dtype=np.float32)
        std = np.array(feature_stats["std"], dtype=np.float32)
        std = np.clip(std, 1e-2, np.inf)
        return mean, std

    action_mean, action_std = _extract(action_key)
    qpos_mean, qpos_std = _extract(qpos_key)

    example_qpos = lerobot_dataset[0][qpos_key].cpu().numpy()

    return {
        "action_mean": action_mean,
        "action_std": action_std,
        "qpos_mean": qpos_mean,
        "qpos_std": qpos_std,
        "example_qpos": example_qpos,
    }


def load_data_lerobot(
    repo_id,
    episode_len,
    batch_size_train,
    batch_size_val,
    *,
    camera_names=None,
    qpos_key="observation.state",
    action_key="action",
    root=None,
    episode_indices=None,
    num_episodes=None,
    train_ratio=0.8,
    seed=0,
    is_sim=False,
    force_cache_sync=False,
    download_videos=True,
    delta_timestamps=None,
    tolerance_s=1e-4,
    num_workers=0,
    pin_memory=True,
    lerobot_kwargs=None,
):
    """
    Load a LeRobot dataset and adapt it to the ACT training pipeline.
    """
    if episode_len <= 0:
        raise ValueError("Argument 'episode_len' must be > 0.")
    extra_kwargs = lerobot_kwargs.copy() if lerobot_kwargs else {}
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=episode_indices,
        delta_timestamps=delta_timestamps,
        tolerance_s=tolerance_s,
        force_cache_sync=force_cache_sync,
        download_videos=download_videos,
        **extra_kwargs,
    )

    if episode_indices is not None:
        available_episode_ids = [int(ep) for ep in episode_indices]
    else:
        available_episode_ids = list(range(dataset.num_episodes))
    if num_episodes is not None:
        available_episode_ids = available_episode_ids[:num_episodes]
    if len(available_episode_ids) == 0:
        raise ValueError("No episodes available after applying filters.")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(available_episode_ids)
    split_idx = int(train_ratio * len(shuffled))
    split_idx = min(max(split_idx, 1), len(shuffled) - 1) if len(shuffled) > 1 else 1
    train_ids = shuffled[:split_idx]
    val_ids = shuffled[split_idx:] if len(shuffled) > 1 else shuffled[:1]
    train_ids = [int(ep) for ep in train_ids]
    val_ids = [int(ep) for ep in val_ids]

    resolved_cameras = _resolve_camera_keys(camera_names, dataset.meta.camera_keys)
    norm_stats = _get_lerobot_norm_stats(dataset, qpos_key, action_key)

    train_dataset = LeRobotEpisodicDataset(
        dataset, train_ids, resolved_cameras, norm_stats, episode_len, qpos_key, action_key
    )
    val_dataset = LeRobotEpisodicDataset(
        dataset, val_ids, resolved_cameras, norm_stats, episode_len, qpos_key, action_key
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=pin_memory, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=pin_memory, num_workers=num_workers
    )

    return train_dataloader, val_dataloader, norm_stats, is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
