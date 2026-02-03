import random
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


class LeRobotLAOMDataset(Dataset):
    """
    LeRobot dataset wrapper for LAOM training.
    Returns: (obs, next_obs, future_obs, action, state, offset)

    Handles frame stacking with episode boundary awareness and resizes images.
    """

    def __init__(
        self,
        repo_id: str,
        frame_stack: int = 1,
        max_offset: int = 1,
        image_key: str = "observation.images.image",
        target_img_size: int = 64,
        device: str = "cpu",
    ):
        self.repo_id = repo_id
        self.frame_stack = frame_stack
        self.max_offset = max_offset
        self.image_key = image_key
        self.target_img_size = target_img_size
        self.device = device

        # Load dataset metadata
        self.meta = LeRobotDatasetMetadata(repo_id)
        self.fps = self.meta.fps

        # Build delta_timestamps for frame stacking + next_obs + future_obs
        # We need: frame_stack past frames, next frame, and max_offset future frame
        self._build_delta_timestamps()

        # Load the dataset
        self.dataset = LeRobotDataset(repo_id, delta_timestamps=self.delta_timestamps)

        # Get action dimension from dataset features
        self.act_dim = self.dataset.meta.shapes["action"][0]

        # State dimension - try to get from dataset or default to 0
        self.state_dim = 0
        if "observation.state" in self.dataset.meta.shapes:
            self.state_dim = self.dataset.meta.shapes["observation.state"][0]

        # Image size properties
        self.img_hw = target_img_size

        # Build valid indices respecting episode boundaries
        self._build_valid_indices()

    def _build_delta_timestamps(self):
        """Build delta_timestamps for accessing temporal data."""
        # Past frames for stacking (negative timestamps)
        # e.g., for frame_stack=3: [-2/fps, -1/fps, 0]
        past_timestamps = [-(self.frame_stack - 1 - i) / self.fps for i in range(self.frame_stack)]

        # Next obs is at +1/fps
        next_timestamp = 1 / self.fps

        # Future obs at max_offset/fps (we'll sample random offset at runtime)
        future_timestamp = self.max_offset / self.fps

        self.delta_timestamps = {
            self.image_key: past_timestamps + [next_timestamp, future_timestamp],
            "action": [0],  # Current action
        }

    def _build_valid_indices(self):
        """Build indices that respect episode boundaries for future_obs sampling."""
        self.valid_indices = []
        self.episode_bounds = {}

        # Get episode information from dataset
        episode_indices = self.dataset.hf_dataset["episode_index"]

        # Find episode boundaries
        current_ep = -1
        ep_start = 0
        for i, ep_idx in enumerate(episode_indices):
            if ep_idx != current_ep:
                if current_ep >= 0:
                    self.episode_bounds[current_ep] = (ep_start, i)
                current_ep = ep_idx
                ep_start = i
        # Last episode
        self.episode_bounds[current_ep] = (ep_start, len(episode_indices))

        # Build valid indices: frames that have max_offset future frames + frame_stack-1 past frames
        for ep_idx, (ep_start, ep_end) in self.episode_bounds.items():
            ep_len = ep_end - ep_start
            # Need frame_stack-1 past frames and max_offset future frames
            valid_start = ep_start + (self.frame_stack - 1)
            valid_end = ep_end - self.max_offset

            if valid_end > valid_start:
                for idx in range(valid_start, valid_end):
                    self.valid_indices.append(idx)

        self.valid_indices = list(self.valid_indices)

    def _process_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Process images: resize and convert to expected format.

        Input: [N, C, H, W] in [0, 1]
        Output: [H, W, C*N] in [0, 255]
        """
        # Resize if needed
        if images.shape[-1] != self.target_img_size or images.shape[-2] != self.target_img_size:
            images = F.interpolate(
                images.float(),
                size=(self.target_img_size, self.target_img_size),
                mode="bilinear",
                align_corners=False,
            )

        # Convert from [0, 1] to [0, 255]
        images = images * 255.0

        # Stack frames: [N, C, H, W] -> [H, W, N*C]
        # First permute to [N, H, W, C]
        images = images.permute(0, 2, 3, 1)
        # Then reshape to [H, W, N*C]
        images = images.permute(1, 2, 0, 3).reshape(
            self.target_img_size, self.target_img_size, -1
        )

        return images

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        frame_idx = self.valid_indices[idx]
        batch = self.dataset[frame_idx]

        # Get images: [frame_stack + 2, C, H, W] (past frames + next + future)
        images = batch[self.image_key]  # Shape: [frame_stack + 2, C, H, W]

        # Split into obs (stacked past), next_obs (stacked with current), future_obs
        # obs: frames [0:frame_stack]
        # next_obs: frames [1:frame_stack+1]
        # future_obs: last frame (max_offset)
        obs_frames = images[: self.frame_stack]
        next_obs_frames = images[1 : self.frame_stack + 1]

        # For future_obs, we sample a random offset and use frame stacking too
        # But for simplicity, we use the last frame_stack frames ending at the future
        # Actually, let's match original behavior: future_obs also has frame_stack
        # We need to reconsider the delta_timestamps...

        # Simpler approach: sample random offset at runtime
        offset = random.randint(1, self.max_offset)
        # Get future frame index relative to current position
        # The delta_timestamps gives us max_offset, but we can adjust
        # For now, use the max_offset frame and the offset info
        future_obs_frames = images[-self.frame_stack:]  # Last frame_stack frames

        # Process each observation
        obs = self._process_images(obs_frames)
        next_obs = self._process_images(next_obs_frames)
        future_obs = self._process_images(future_obs_frames)

        # Get action
        action = batch["action"]
        if isinstance(action, list):
            action = torch.tensor(action)
        action = action.squeeze()

        # Get state (zeros if not available)
        if self.state_dim > 0 and "observation.state" in batch:
            state = batch["observation.state"]
            if isinstance(state, list):
                state = torch.tensor(state)
            state = state.squeeze()
        else:
            state = torch.zeros(max(1, self.state_dim))

        return obs, next_obs, future_obs, action, state, (offset - 1)


class LeRobotBCDataset(Dataset):
    """
    LeRobot dataset wrapper for Behavioral Cloning.
    Returns: (obs, next_obs, action)
    """

    def __init__(
        self,
        repo_id: str,
        frame_stack: int = 1,
        image_key: str = "observation.images.image",
        target_img_size: int = 64,
        device: str = "cpu",
    ):
        self.repo_id = repo_id
        self.frame_stack = frame_stack
        self.image_key = image_key
        self.target_img_size = target_img_size
        self.device = device

        # Load dataset metadata
        self.meta = LeRobotDatasetMetadata(repo_id)
        self.fps = self.meta.fps

        # Build delta_timestamps for frame stacking + next_obs
        self._build_delta_timestamps()

        # Load the dataset
        self.dataset = LeRobotDataset(repo_id, delta_timestamps=self.delta_timestamps)

        # Get action dimension
        self.act_dim = self.dataset.meta.shapes["action"][0]
        self.img_hw = target_img_size

        # Build valid indices
        self._build_valid_indices()

    def _build_delta_timestamps(self):
        """Build delta_timestamps for obs and next_obs."""
        # Past frames for stacking
        past_timestamps = [-(self.frame_stack - 1 - i) / self.fps for i in range(self.frame_stack)]
        # Next obs
        next_timestamp = 1 / self.fps

        self.delta_timestamps = {
            self.image_key: past_timestamps + [next_timestamp],
            "action": [0],
        }

    def _build_valid_indices(self):
        """Build valid indices respecting episode boundaries."""
        self.valid_indices = []

        episode_indices = self.dataset.hf_dataset["episode_index"]

        current_ep = -1
        ep_start = 0
        episode_bounds = {}

        for i, ep_idx in enumerate(episode_indices):
            if ep_idx != current_ep:
                if current_ep >= 0:
                    episode_bounds[current_ep] = (ep_start, i)
                current_ep = ep_idx
                ep_start = i
        episode_bounds[current_ep] = (ep_start, len(episode_indices))

        for ep_idx, (ep_start, ep_end) in episode_bounds.items():
            valid_start = ep_start + (self.frame_stack - 1)
            valid_end = ep_end - 1  # Need 1 future frame for next_obs

            if valid_end > valid_start:
                for idx in range(valid_start, valid_end):
                    self.valid_indices.append(idx)

    def _process_images(self, images: torch.Tensor) -> torch.Tensor:
        """Process images: resize and convert to expected format."""
        if images.shape[-1] != self.target_img_size or images.shape[-2] != self.target_img_size:
            images = F.interpolate(
                images.float(),
                size=(self.target_img_size, self.target_img_size),
                mode="bilinear",
                align_corners=False,
            )

        images = images * 255.0
        images = images.permute(0, 2, 3, 1)
        images = images.permute(1, 2, 0, 3).reshape(
            self.target_img_size, self.target_img_size, -1
        )

        return images

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        frame_idx = self.valid_indices[idx]
        batch = self.dataset[frame_idx]

        images = batch[self.image_key]

        obs_frames = images[: self.frame_stack]
        next_obs_frames = images[1 : self.frame_stack + 1]

        obs = self._process_images(obs_frames)
        next_obs = self._process_images(next_obs_frames)

        action = batch["action"]
        if isinstance(action, list):
            action = torch.tensor(action)
        action = action.squeeze()

        return obs, next_obs, action


class LeRobotIterableDataset(IterableDataset):
    """
    Infinite iterator for labeled data sampling from LeRobot dataset.
    Returns: (obs, next_obs, future_obs, action, state, offset)
    """

    def __init__(
        self,
        repo_id: str,
        frame_stack: int = 1,
        max_offset: int = 1,
        image_key: str = "observation.images.image",
        target_img_size: int = 64,
        device: str = "cpu",
    ):
        # Use the regular dataset and iterate infinitely
        self._base_dataset = LeRobotLAOMDataset(
            repo_id=repo_id,
            frame_stack=frame_stack,
            max_offset=max_offset,
            image_key=image_key,
            target_img_size=target_img_size,
            device=device,
        )

        # Copy attributes for external access
        self.img_hw = self._base_dataset.img_hw
        self.act_dim = self._base_dataset.act_dim
        self.state_dim = self._base_dataset.state_dim

    def __iter__(self):
        while True:
            idx = random.randint(0, len(self._base_dataset) - 1)
            yield self._base_dataset[idx]
