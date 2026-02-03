import math
import time
import uuid
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

from src.augmentations import Augmenter
from src.lerobot_dataset import LeRobotLAOMDataset, LeRobotBCDataset
from src.nn import Actor, IDMLabels
from src.scheduler import linear_annealing_with_warmup
from src.utils import (
    get_grad_norm,
    get_optim_groups,
    normalize_img,
    set_seed,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DatasetConfig:
    repo_id: str = "HuggingFaceVLA/libero"
    image_key: str = "observation.images.image"
    target_img_size: int = 64


@dataclass
class IDMConfig:
    total_updates: int = 2500
    batch_size: int = 256
    use_aug: bool = True
    future_obs_offset: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    grad_norm: Optional[float] = None
    act_head_dim: int = 512
    act_head_dropout: float = 0.0
    encoder_scale: int = 1
    encoder_num_res_blocks: int = 1
    encoder_deep: bool = True
    encoder_dropout: float = 0.0
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
class Config:
    project: str = "laom"
    group: str = "idm"
    name: str = "idm"
    seed: int = 0

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    idm: IDMConfig = field(default_factory=IDMConfig)
    bc: BCConfig = field(default_factory=BCConfig)

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())}"


def train_idm(config: IDMConfig, dataset_config: DatasetConfig):
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
    )
    num_epochs = max(1, config.total_updates // len(dataloader))

    img_size = dataset_config.target_img_size
    idm = IDMLabels(
        shape=(3 * config.frame_stack, img_size, img_size),
        act_dim=dataset.act_dim,
        act_head_dim=config.act_head_dim,
        act_head_dropout=config.act_head_dropout,
        encoder_scale=config.encoder_scale,
        encoder_channels=(16, 32, 64, 128, 256) if config.encoder_deep else (16, 32, 32),
        encoder_num_res_blocks=config.encoder_num_res_blocks,
        encoder_dropout=config.encoder_dropout,
    ).to(DEVICE)

    torchinfo.summary(
        idm,
        input_size=[
            (1, 3 * config.frame_stack, img_size, img_size),
            (1, 3 * config.frame_stack, img_size, img_size),
        ],
    )
    optim = torch.optim.Adam(
        params=get_optim_groups(idm, config.weight_decay),
        lr=config.learning_rate,
        fused=True,
    )
    augmenter = Augmenter(img_size)

    # State probe uses zeros since LeRobot may not have proprioceptive state
    state_dim = max(1, dataset.state_dim)
    state_probe = nn.Linear(math.prod(idm.final_encoder_shape), state_dim).to(DEVICE)
    state_probe_optim = torch.optim.Adam(state_probe.parameters(), lr=config.learning_rate)

    # scheduler setup
    total_updates = len(dataloader) * num_epochs
    warmup_updates = len(dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)

    start_time = time.time()
    total_steps = 0
    total_tokens = 0
    for epoch in trange(num_epochs, desc="Epochs"):
        idm.train()
        for i, batch in enumerate(dataloader):
            total_tokens += config.batch_size
            total_steps += 1

            obs, _, future_obs, actions, states, _ = [b.to(DEVICE) for b in batch]
            obs = normalize_img(obs.permute((0, 3, 1, 2)))
            future_obs = normalize_img(future_obs.permute((0, 3, 1, 2)))

            if config.use_aug:
                obs = augmenter(obs)
                future_obs = augmenter(future_obs)

            # update idm
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_action, obs_emb = idm(obs, future_obs)
                loss = F.mse_loss(pred_action, actions)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(idm.parameters(), max_norm=config.grad_norm)
            optim.step()
            scheduler.step()

            # evaluation
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_states = state_probe(obs_emb.detach())
                state_probe_loss = F.mse_loss(pred_states, states)

            state_probe_optim.zero_grad(set_to_none=True)
            state_probe_loss.backward()
            state_probe_optim.step()

            wandb.log(
                {
                    "idm/mse_loss": loss.item(),
                    "idm/state_probe_loss": state_probe_loss.item(),
                    "idm/throughput": total_tokens / (time.time() - start_time),
                    "idm/learning_rate": scheduler.get_last_lr()[0],
                    "idm/grad_norm": get_grad_norm(idm).item(),
                    "idm/obs_hidden_norm": torch.norm(obs_emb, p=2, dim=-1).mean().item(),
                    "idm/epoch": epoch,
                    "idm/total_steps": total_steps,
                }
            )

    return idm


def train_bc(lam: IDMLabels, config: BCConfig, dataset_config: DatasetConfig):
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
    num_actions = dataset.act_dim
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
    ).to(DEVICE)

    optim = torch.optim.AdamW(params=get_optim_groups(actor, config.weight_decay), lr=config.learning_rate, fused=True)
    # scheduler setup
    total_updates = len(dataloader) * config.num_epochs
    warmup_updates = len(dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)

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

            obs, next_obs, debug_true_actions = [b.to(DEVICE) for b in batch]
            # rescale from 0..255 -> -1..1
            obs = normalize_img(obs.permute((0, 3, 1, 2)))
            next_obs = normalize_img(next_obs.permute((0, 3, 1, 2)))

            # label with idm latent actions
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

            wandb.log(
                {
                    "bc/mse_loss": loss.item(),
                    "bc/throughput": total_tokens / (time.time() - start_time),
                    "bc/learning_rate": scheduler.get_last_lr()[0],
                    "bc/epoch": epoch,
                    "bc/total_steps": total_steps,
                }
            )

    return actor


@pyrallis.wrap()
def train(config: Config):
    run = wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        save_code=True,
    )
    print(config)

    set_seed(config.seed)
    # stage 1: pretraining idm on unlabeled dataset
    idm = train_idm(config=config.idm, dataset_config=config.dataset)
    # stage 2: pretraining bc on idm labeled actions
    actor = train_bc(lam=idm, config=config.bc, dataset_config=config.dataset)

    run.finish()
    return idm, actor


if __name__ == "__main__":
    train()
