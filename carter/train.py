from tqdm import tqdm
from pathlib import Path

import torch
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import TrainConfig, pgn_collator


class Trainer:
    def __init__(
        self,
        config: TrainConfig,
        model,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset,
    ):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # determine the device we'll train on
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        if self.config.verbose:
            print("running on device", self.device)
        self.save_folder = Path(self.config.save_folder)
        self.save_folder.mkdir(parents=True, exist_ok=config.folder_exist_ok)

    def autoregressive_loss(self, model, batch) -> torch.Tensor:
        raise NotImplementedError
        pgn, boards, move_idx = (
            batch["pgn"].to(self.device),
            batch["boards"].to(self.device),
            batch["move_idx"].to(self.device),
        )
        logits = model(pgn).logits
        # TODO!!!

    def prediction_loss(self, model, batch) -> torch.Tensor:
        pgn, boards, move_idx = (
            batch["pgn"].to(self.device),
            batch["boards"].to(self.device),
            batch["move_idx"].to(self.device),
        )
        weights = torch.ones_like(boards)
        pgn[pgn == -100] = 0
        weights[boards == -100] = 0
        pgn[pgn == -100] = self.config.char_emb["\n"]
        logits = model(pgn).logits
        idx = move_idx.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
        pred_logits = torch.gather(logits, 1, idx.to(torch.int64))
        loss = F.binary_cross_entropy_with_logits(
            pred_logits, boards.float(), weight=weights, reduction="sum"
        )
        return loss / weights.sum()

    def run(self):
        model, config = self.model.to(self.device), self.config

        # setup the optimizer
        optimizer = torch.optim.AdamW(
            model.lm_head.parameters() if config.optim_head else model.parameters(),
            lr=config.learning_rate,
        )

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=pgn_collator,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        eval_loader = DataLoader(
            self.eval_dataset,
            shuffle=False,
            collate_fn=pgn_collator,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        logs = {"train": [], "eval": []}

        for epoch in tqdm(range(1, config.epochs + 1)):
            print(f"Epoch: {epoch}")
            model.train()
            train_losses = []
            for batch in train_loader:
                # forward the model
                if config.train_type == "autoregressive":
                    raise NotImplementedError
                elif config.train_type == "prediction":
                    loss = self.prediction_loss(model, batch)
                else:
                    raise Exception(
                        f"config.train_type is invalid: {config.train_type}"
                    )

                # backprop and update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logs["train"].append(loss.item())
                train_losses.append(loss.item())
            print(f"Train loss: {torch.mean(torch.tensor(train_losses)):.4f}")

            model.eval()
            with torch.no_grad():
                eval_losses = []
                for batch in eval_loader:
                    # forward the model
                    if config.train_type == "autoregressive":
                        raise NotImplementedError
                    elif config.train_type == "prediction":
                        loss = self.prediction_loss(model, batch)
                    else:
                        raise Exception(
                            f"config.train_type is invalid: {config.train_type}"
                        )
                    eval_losses.append(loss.item())
                logs["eval"].append(torch.mean(torch.tensor(eval_losses)).item())
                print(f"Eval loss: {logs['eval'][-1]:.4f}")

            if config.save_per_epoch or epoch == self.config.epochs:
                torch.save(model.state_dict(), self.save_folder / f"epoch_{epoch}.pt")
        with open(self.save_folder / "logs", "w") as f:
            f.write(str(self.config) + "\n")
            f.write("train losses:")
            for loss in logs["train"]:
                f.write(str(loss) + "\n")
            f.write("eval losses:")
            for loss in logs["eval"]:
                f.write(str(loss) + "\n")
