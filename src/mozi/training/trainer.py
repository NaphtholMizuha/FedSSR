import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple
from copy import deepcopy

StateDict = Dict[str, torch.Tensor]
StateTemplate = Dict[str, Tuple[int, ...]]



class Trainer:
    def __init__(
        self,
        model: nn.Module,
        init_state: dict,
        train_set: Dataset,
        test_set: Dataset,
        bs: int,
        nw: int,
        lr: float,
        device: str,
    ) -> None:
        self.model = model
        self.model.load_state_dict(init_state)
        self.shapes = {key: value.shape for key, value in init_state.items()}
        self.train_loader = DataLoader(
            train_set, batch_size=bs, num_workers=nw, shuffle=True
        )
        self.test_loader = DataLoader(test_set, batch_size=bs, num_workers=nw)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=lr)
        self.device = device
        self.lr = lr
        self.state = self.flat(deepcopy(init_state)).to(device)

    def train(self):
        self.model.train()
        loss_sum = 0.0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            loss_sum += loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss_sum / len(self.train_loader)

    def local_train(self, n_epoch):
        for _ in range(n_epoch):
            self.train()

    def get_grad(self):
        return self.flat(self.model.state_dict()) - self.state

    def set_grad(self, grad):
        self.state += grad
        new_state = self.unflat(self.state, self.shapes)
        self.model.load_state_dict(new_state)

    def test(self, dataloader=None):
        self.model.eval()
        criterion = nn.CrossEntropyLoss().to(self.device)

        loss, acc = 0, 0

        with torch.no_grad():
            dataloader = dataloader or self.test_loader

            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss += criterion(pred, y).item()
                acc += (pred.argmax(1) == y).type(torch.float).sum().item()

        loss /= len(self.test_loader)
        acc /= len(self.test_loader.dataset)
        return loss, acc

    @staticmethod
    def flat(x: StateDict) -> torch.Tensor:
        w_list = []
        for weight in x.values():
            if weight.shape != torch.Size([]):
                w_i = weight.flatten()
                w_list.append(w_i)
        return torch.cat(w_list)

    def unflat(self, x: torch.Tensor, shapes: StateTemplate) -> StateDict:
        state = {}
        start_idx = 0

        for key, shape in shapes.items():
            # print(f"key: {key}, shape: {shape}")
            if len(shape) != 0:
                size = torch.prod(torch.tensor(shape))
                # print(size)
                slice = x[start_idx : start_idx + size].reshape(shape).to(self.device)
                state[key] = slice
                start_idx += size

        return state
