import os, yaml
import math
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Optional

import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
import wandb
from pyrallis import field
from torch.utils.data import DataLoader
from tqdm import trange
import tqdm

from src.augmentations import Augmenter
from src.lerobot_dataset import LeRobotLAOMDataset, LeRobotBCDataset, LeRobotIterableDataset
from src.nn import ActionDecoder, Actor, LAOMWithLabels
from src.scheduler import linear_annealing_with_warmup
from src.utils import (
    get_grad_norm,
    get_optim_groups,
    set_seed,
    soft_update,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# From: https://github.com/joon-stack/laom/blob/main/train_laom_labels.py
def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath, config=None):
    """Utility to save a model with training information."""
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    
    # Save the config into a YAML file
    if config is not None:
        config_filepath = filepath.replace('.pt', '_config.yaml')
        with open(config_filepath, 'w') as f:
            yaml.dump(asdict(config), f, default_flow_style=False, allow_unicode=True)
        print(f"Config saved: {config_filepath}")
    
    torch.save(checkpoint_data, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, scheduler, filepath):
    """Loads a model and the training information."""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=DEVICE)
        
        # Debugging: print shapes before loading
        if 'model_state_dict' in checkpoint and hasattr(model, 'true_actions_head'):
            print("--- Shape Debug ---")
            # Shape in the current model
            current_shape = model.true_actions_head.weight.shape
            print(f"Current model's 'true_actions_head' shape: {current_shape}")
            
            # Shape in the checkpoint
            checkpoint_shape = checkpoint['model_state_dict']['true_actions_head.weight'].shape
            print(f"Checkpoint's 'true_actions_head' shape: {checkpoint_shape}")
            print("-------------------")

        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded checkpoint: {filepath} (epoch {checkpoint['epoch']})")
        
        # Load the config from the YAML file.
        config_filepath = filepath.replace('.pt', '_config.yaml')
        if os.path.exists(config_filepath):
            with open(config_filepath, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Config loaded: {config_filepath}")
            print("Keys loaded from the config:")
            for key, value in config.items():
                print(f"  {key}: {value}")
            return checkpoint['epoch'], checkpoint['loss'], config
        
        return checkpoint['epoch'], checkpoint['loss'], None
    return 0, float('inf'), None


def prepare_obs(img: torch.Tensor, target_size: int) -> torch.Tensor:
    """Resize and normalize images on GPU.

    Input: [B, C, H, W] in [0, 1]
    Output: [B, C, target_size, target_size] in [-1, 1]
    """
    if img.shape[-1] != target_size or img.shape[-2] != target_size:
        img = F.interpolate(img, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return img * 2.0 - 1.0


@dataclass
class DatasetConfig:
    repo_id: str = "HuggingFaceVLA/libero"
    image_key: str = "observation.images.image"
    target_img_size: int = 64


@dataclass
class LAOMConfig:
    num_epochs: int = 10
    batch_size: int = 512
    labeled_batch_size: int = 512
    labeled_loss_coef: float = 0.05
    cosine_loss: bool = False
    use_aug: bool = False
    future_obs_offset: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    grad_norm: Optional[float] = None
    latent_action_dim: int = 256
    act_head_dim: int = 512
    act_head_dropout: float = 0.0
    obs_head_dim: int = 512
    obs_head_dropout: float = 0.0
    encoder_scale: int = 1
    encoder_num_res_blocks: int = 1
    encoder_dropout: float = 0.0
    encoder_norm_out: bool = True
    encoder_deep: bool = True
    target_tau: float = 0.01
    target_update_every: int = 1
    frame_stack: int = 3


@dataclass
class BCConfig:
    num_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    encoder_scale: int = 1
    encoder_num_res_blocks: int = 2
    encoder_deep: bool = False
    dropout: float = 0.0
    use_aug: bool = True
    frame_stack: int = 3


@dataclass
class DecoderConfig:
    total_updates: int = 1
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    hidden_dim: int = 128
    use_aug: bool = True


@dataclass
class Config:
    project: str = "laom"
    group: str = "laom-labels"
    name: str = "laom-labels"
    seed: int = 0

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    lapo: LAOMConfig = field(default_factory=LAOMConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())}"


def train_laom(config: LAOMConfig, dataset_config: DatasetConfig, checkpoint_dir: str = "/app/checkpoints/laom/"):
    dataset = LeRobotLAOMDataset(
        repo_id=dataset_config.repo_id,
        frame_stack=config.frame_stack,
        max_offset=config.future_obs_offset,
        image_key=dataset_config.image_key,
        target_img_size=dataset_config.target_img_size,
        device=DEVICE,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # For labeled data, use the same dataset with iterable wrapper
    labeled_dataset = LeRobotIterableDataset(
        repo_id=dataset_config.repo_id,
        frame_stack=config.frame_stack,
        max_offset=config.future_obs_offset,
        image_key=dataset_config.image_key,
        target_img_size=dataset_config.target_img_size,
        device=DEVICE,
    )
    labeled_dataloader = DataLoader(
        labeled_dataset,
        batch_size=config.labeled_batch_size,
        num_workers=4,
        pin_memory=True,
    )

    img_size = dataset_config.target_img_size
    lapo = LAOMWithLabels(
        shape=(3 * config.frame_stack, img_size, img_size),
        true_act_dim=dataset.act_dim,
        latent_act_dim=config.latent_action_dim,
        act_head_dim=config.act_head_dim,
        act_head_dropout=config.act_head_dropout,
        obs_head_dim=config.obs_head_dim,
        obs_head_dropout=config.obs_head_dropout,
        encoder_scale=config.encoder_scale,
        encoder_channels=(16, 32, 64, 128, 256) if config.encoder_deep else (16, 32, 32),
        encoder_num_res_blocks=config.encoder_num_res_blocks,
        encoder_dropout=config.encoder_dropout,
        encoder_norm_out=config.encoder_norm_out,
    ).to(DEVICE, non_blocking=True)

    target_lapo = deepcopy(lapo)
    for p in target_lapo.parameters():
        p.requires_grad_(False)

    torchinfo.summary(
        lapo,
        input_size=[
            (1, 3 * config.frame_stack, img_size, img_size),
            (1, 3 * config.frame_stack, img_size, img_size),
        ],
    )
    optim = torch.optim.Adam(
        params=get_optim_groups(lapo, config.weight_decay),
        lr=config.learning_rate,
        fused=True,
    )
    augmenter = Augmenter(img_size)

    # State probe uses zeros since LeRobot may not have proprioceptive state
    state_dim = max(1, dataset.state_dim)
    state_probe = nn.Linear(math.prod(lapo.final_encoder_shape), state_dim).to(DEVICE, non_blocking=True)
    state_probe_optim = torch.optim.Adam(state_probe.parameters(), lr=config.learning_rate)

    act_linear_probe = nn.Linear(config.latent_action_dim, dataset.act_dim).to(DEVICE, non_blocking=True)
    act_probe_optim = torch.optim.Adam(act_linear_probe.parameters(), lr=config.learning_rate)

    print("Final encoder shape:", math.prod(lapo.final_encoder_shape))
    state_act_linear_probe = nn.Linear(math.prod(lapo.final_encoder_shape), dataset.act_dim).to(DEVICE, non_blocking=True)
    state_act_probe_optim = torch.optim.Adam(state_act_linear_probe.parameters(), lr=config.learning_rate)

    # scheduler setup
    total_updates = len(dataloader) * config.num_epochs
    warmup_updates = len(dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)

    start_time = time.time()
    total_iterations = 0
    total_tokens = 0

    labeled_dataloader_iter = iter(labeled_dataloader)
    for epoch in range(config.num_epochs):
        lapo.train()
        for i, batch in tqdm.tqdm(enumerate(dataloader), desc=f"Epoch #{epoch}", total=len(dataloader)):
            total_tokens += config.batch_size
            total_iterations += 1

            obs, next_obs, future_obs, debug_actions, debug_states, _ = [b.to(DEVICE, non_blocking=True) for b in batch]

            obs = prepare_obs(obs, img_size)
            next_obs = prepare_obs(next_obs, img_size)
            future_obs = prepare_obs(future_obs, img_size)

            if config.use_aug:
                obs_aug = augmenter(obs)
                future_obs_aug = augmenter(future_obs)
                next_obs_aug = augmenter(next_obs)

            # update lapo
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                if config.use_aug:
                    latent_next_obs, latent_action, obs_hidden = lapo(obs_aug, future_obs_aug)
                else:
                    latent_next_obs, latent_action, obs_hidden = lapo(obs, future_obs)

                with torch.no_grad():
                    if config.use_aug:
                        next_obs_target = target_lapo.encoder(next_obs_aug).flatten(1)
                    else:
                        next_obs_target = target_lapo.encoder(next_obs).flatten(1)

                if config.cosine_loss:
                    loss0 = 1 - F.cosine_similarity(latent_next_obs, next_obs_target.detach(), dim=-1).mean()
                else:
                    loss0 = F.mse_loss(latent_next_obs, next_obs_target.detach())

            # loss with true actions
            labeled_batch = next(labeled_dataloader_iter)
            label_obs, label_next_obs, label_future_obs, label_actions, _, _ = [b.to(DEVICE, non_blocking=True) for b in labeled_batch]

            label_obs = prepare_obs(label_obs, img_size)
            label_future_obs = prepare_obs(label_future_obs, img_size)
            label_next_obs = prepare_obs(label_next_obs, img_size)

            if config.use_aug:
                label_obs_aug = augmenter(label_obs)
                label_future_obs_aug = augmenter(label_future_obs)

            # update lapo
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                if config.use_aug:
                    _, _, pred_action, _ = lapo(label_obs_aug, label_future_obs_aug, predict_true_act=True)
                else:
                    _, _, pred_action, _ = lapo(label_obs, label_future_obs, predict_true_act=True)

                loss1 = F.mse_loss(pred_action, label_actions)

            loss = loss0 + config.labeled_loss_coef * loss1

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(lapo.parameters(), max_norm=config.grad_norm)
            optim.step()
            scheduler.step()
            if i % config.target_update_every == 0:
                soft_update(target_lapo, lapo, tau=config.target_tau)

            # update state probe
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_states = state_probe(obs_hidden.detach())
                state_probe_loss = F.mse_loss(pred_states, debug_states)

            state_probe_optim.zero_grad(set_to_none=True)
            state_probe_loss.backward()
            state_probe_optim.step()

            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_action = act_linear_probe(latent_action.detach())
                act_probe_loss = F.mse_loss(pred_action, debug_actions)

            act_probe_optim.zero_grad(set_to_none=True)
            act_probe_loss.backward()
            act_probe_optim.step()

            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                state_pred_action = state_act_linear_probe(obs_hidden.detach())
                state_act_probe_loss = F.mse_loss(state_pred_action, debug_actions)

            state_act_probe_optim.zero_grad(set_to_none=True)
            state_act_probe_loss.backward()
            state_act_probe_optim.step()

            if total_iterations % 20 == 0:
                wandb.log(
                    {
                        "lapo/total_loss": loss.item(),
                        "lapo/mse_loss": loss0.item(),
                        "lapo/true_action_mse_loss": loss1.item(),
                        "lapo/state_probe_mse_loss": state_probe_loss.item(),
                        "lapo/action_probe_mse_loss": act_probe_loss.item(),
                        "lapo/state_action_probe_mse_loss": state_act_probe_loss.item(),
                        "lapo/throughput": total_tokens / (time.time() - start_time),
                        "lapo/learning_rate": scheduler.get_last_lr()[0],
                        "lapo/grad_norm": get_grad_norm(lapo).item(),
                        "lapo/target_obs_norm": torch.norm(next_obs_target, p=2, dim=-1).mean().item(),
                        "lapo/online_obs_norm": torch.norm(latent_next_obs, p=2, dim=-1).mean().item(),
                        "lapo/latent_act_norm": torch.norm(latent_action, p=2, dim=-1).mean().item(),
                        "lapo/epoch": epoch,
                        "lapo/total_steps": total_iterations,
                    }
                )

        # Save LAOM every epoch.
        save_checkpoint(
            lapo,
            optim,
            scheduler,
            config.num_epochs - 1,
            loss.item(),
            os.path.join(checkpoint_dir, "lapo_final.pt"),
            config,
        )


    return lapo


def train_bc(lam: LAOMWithLabels, config: BCConfig, dataset_config: DatasetConfig):
    dataset = LeRobotBCDataset(
        repo_id=dataset_config.repo_id,
        frame_stack=config.frame_stack,
        image_key=dataset_config.image_key,
        target_img_size=dataset_config.target_img_size,
        device=DEVICE,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    img_size = dataset_config.target_img_size
    num_actions = lam.latent_act_dim
    for p in lam.parameters():
        p.requires_grad_(False)
    lam.eval()

    actor = Actor(
        shape=(3 * config.frame_stack, img_size, img_size),
        num_actions=num_actions,
        encoder_scale=config.encoder_scale,
        encoder_channels=(16, 32, 64, 128, 256) if config.encoder_deep else (16, 32, 32),
        encoder_num_res_blocks=config.encoder_num_res_blocks,
        dropout=config.dropout,
    ).to(DEVICE, non_blocking=True)

    optim = torch.optim.AdamW(params=get_optim_groups(actor, config.weight_decay), lr=config.learning_rate, fused=True)
    # scheduler setup
    total_updates = len(dataloader) * config.num_epochs
    warmup_updates = len(dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)

    # for debug
    print("Latent action dim:", num_actions)
    act_decoder = nn.Sequential(
        nn.Linear(num_actions, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, dataset.act_dim)
    ).to(DEVICE, non_blocking=True)

    act_decoder_optim = torch.optim.AdamW(params=act_decoder.parameters(), lr=config.learning_rate, fused=True)
    act_decoder_scheduler = linear_annealing_with_warmup(act_decoder_optim, warmup_updates, total_updates)

    torchinfo.summary(actor, input_size=(1, 3 * config.frame_stack, img_size, img_size))
    if config.use_aug:
        augmenter = Augmenter(img_resolution=img_size)

    start_time = time.time()
    total_tokens = 0
    total_steps = 0
    for epoch in trange(config.num_epochs, desc="Epochs"):
        actor.train()
        for batch in dataloader:
            total_tokens += config.batch_size
            total_steps += 1

            obs, next_obs, true_actions = [b.to(DEVICE, non_blocking=True) for b in batch]
            obs = prepare_obs(obs, img_size)
            next_obs = prepare_obs(next_obs, img_size)

            # label with lapo latent actions
            target_actions = lam.label(obs, next_obs)

            # augment obs only for bc to make action labels determenistic
            if config.use_aug:
                obs = augmenter(obs)

            # update actor
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_actions, _ = actor(obs)
                loss = F.mse_loss(pred_actions, target_actions)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()

            # optimizing the probe
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_true_actions = act_decoder(pred_actions.detach())
                decoder_loss = F.mse_loss(pred_true_actions, true_actions)

            act_decoder_optim.zero_grad(set_to_none=True)
            decoder_loss.backward()
            act_decoder_optim.step()
            act_decoder_scheduler.step()

            if total_steps % 20 == 0:
                wandb.log(
                    {
                        "bc/mse_loss": loss.item(),
                        "bc/throughput": total_tokens / (time.time() - start_time),
                        "bc/learning_rate": scheduler.get_last_lr()[0],
                        "bc/act_decoder_probe_mse_loss": decoder_loss.item(),
                        "bc/epoch": epoch,
                        "bc/total_steps": total_steps,
                    }
                )

    return actor


def train_act_decoder(actor: Actor, config: DecoderConfig, bc_config: BCConfig, dataset_config: DatasetConfig):
    for p in actor.parameters():
        p.requires_grad_(False)
    actor.eval()

    dataset = LeRobotBCDataset(
        repo_id=dataset_config.repo_id,
        frame_stack=bc_config.frame_stack,
        image_key=dataset_config.image_key,
        target_img_size=dataset_config.target_img_size,
        device=DEVICE,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    # to make equal number of updates for all labeled datasets which vary in size
    num_epochs = max(1, config.total_updates // len(dataloader))

    img_size = dataset_config.target_img_size
    action_decoder = ActionDecoder(
        obs_emb_dim=math.prod(actor.final_encoder_shape),
        latent_act_dim=actor.num_actions,
        true_act_dim=dataset.act_dim,
        hidden_dim=config.hidden_dim,
    ).to(DEVICE, non_blocking=True)

    optim = torch.optim.AdamW(
        params=get_optim_groups(action_decoder, config.weight_decay), lr=config.learning_rate, fused=True
    )

    # scheduler setup
    total_updates = len(dataloader) * num_epochs
    warmup_updates = len(dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)

    if config.use_aug:
        augmenter = Augmenter(img_resolution=img_size)

    start_time = time.time()
    total_tokens = 0
    total_steps = 0

    for epoch in trange(num_epochs, desc="Epochs"):
        for batch in dataloader:
            total_tokens += config.batch_size
            total_steps += 1

            obs, _, true_actions = [b.to(DEVICE, non_blocking=True) for b in batch]
            obs = prepare_obs(obs, img_size)

            if config.use_aug:
                obs = augmenter(obs)

            # update actor
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                with torch.no_grad():
                    latent_actions, obs_emb = actor(obs)
                pred_actions = action_decoder(obs_emb, latent_actions)

                loss = F.mse_loss(pred_actions, true_actions)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()

            if total_steps % 20 == 0:
                wandb.log(
                    {
                        "decoder/mse_loss": loss.item(),
                        "decoder/throughput": total_tokens / (time.time() - start_time),
                        "decoder/learning_rate": scheduler.get_last_lr()[0],
                        "decoder/epoch": epoch,
                        "decoder/total_steps": total_steps,
                    }
                )

    return action_decoder


@pyrallis.wrap()
def train(config: Config):
    run = wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        save_code=True,
    )
    set_seed(config.seed)
    # stage 1: pretraining laom on unlabeled dataset
    lapo = train_laom(config=config.lapo, dataset_config=config.dataset)
    # stage 2: pretraining bc on latent actions
    actor = train_bc(lam=lapo, config=config.bc, dataset_config=config.dataset)
    # stage 3: finetune on labeled ground-truth actions
    action_decoder = train_act_decoder(
        actor=actor, config=config.decoder, bc_config=config.bc, dataset_config=config.dataset
    )

    run.finish()
    return lapo, actor, action_decoder


if __name__ == "__main__":
    train()
