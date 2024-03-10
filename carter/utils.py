from dataclasses import dataclass
from typing import List, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel


class PGNDataset(Dataset):
    """Dataset for PGN strings."""

    def __init__(self, data: List[Dict[str, torch.Tensor]]) -> None:
        """Constructor.

        Args:
            - data: list of dicts of data, from process_data.py
        """
        self.data = data
        assert False, "figure out same length please"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        return self.data[index]


@dataclass
class TrainConfig:
    save_folder: str
    train_type: str
    char_emb: Dict[str, int]
    folder_exist_ok: bool = False
    optim_head: bool = False
    verbose: bool = True
    save_per_epoch: bool = True
    # device to train on
    device: str = "auto"
    # dataloder parameters
    num_workers: int = 4
    # optimizer parameters
    epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 3e-4


def create_autoregressive_model(config, verbose=True):
    model = GPT2LMHeadModel(config)
    if verbose:
        print(f"Parameter count: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model


def convert_to_predict(model, config):
    model.lm_head = nn.Sequential(
        nn.Linear(config.n_embd, config.n_embd),
        nn.GELU(),
        nn.Linear(config.n_embd, 8 * 8 * 13),
    )


def convert_to_autoregressive(model, config):
    model.lm_head = nn.Sequential(
        nn.Linear(config.n_embd, config.n_embd),
        nn.GELU(),
        nn.Linear(config.n_embd, config.vocab_size),
    )
