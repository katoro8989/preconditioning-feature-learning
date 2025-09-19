# Standard library
import os
import time
import datetime
import uuid
from pathlib import Path
from dataclasses import dataclass

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from opt import build_optimizer, OptimizerSetting
from utils.eval import eval_single_dataset

@dataclass
class Config:
    optimizer: str = "vanilla_sgd"
    lr: float = 1e-2
    weight_decay: float = 1e-4
    momentum: float = 0.9
    eps: float = 1e-8
    beta_1: float = 0.9
    beta_2: float = 0.999
    rho: float = 0.05
    damping: float = 0.001
    epsilon: float = 1e-8
    update_freq: int = 1
    hessian_power: float = 1.0
    seed: int = 42
    total_epochs: int = 500
    use_scheduler: bool = False
    mode: str = "high"


def trainer(config):
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.backends.cudnn.benchmark = False

    class AddNoiseByLabelDataset(Dataset):
        def __init__(self, dataset, class_noise, noise_map=None):
            self.dataset = dataset
            self.class_noise = class_noise
            self.noise_map = noise_map

        def __getitem__(self, index):
            image, label = self.dataset[index]

            if self.noise_map:
                noise_key = self.noise_map[label]
            else:
                noise_key = label

            noise = self.class_noise[noise_key]
            noisy_image = torch.clamp(image + noise, 0., 1.)

            return noisy_image, label

        def __len__(self):
            return len(self.dataset)

    NOISE_MEAN = 0.
    NOISE_STD = 0.1

    main_class_noise = {
        i: torch.randn(1, 28, 28) * NOISE_STD + NOISE_MEAN
        for i in range(10)
    }

    flip_map = {i: i + 1 if i % 2 == 0 else i - 1 for i in range(10)}

    transform_to_tensor = transforms.Compose([transforms.ToTensor()])

    base_train_dataset = datasets.MNIST(root='./data', train=True, transform=transform_to_tensor, download=True)

    train_dataset = AddNoiseByLabelDataset(base_train_dataset, class_noise=main_class_noise, noise_map=None)

    base_test_dataset = datasets.MNIST(root='./data', train=False, transform=transform_to_tensor, download=True)

    perm = torch.randperm(len(base_test_dataset))[: int(1*len(base_test_dataset))]
    base_test_dataset.targets[perm] = torch.randint(0,10,(len(perm),))

    if config.mode == "high":
        test_dataset = AddNoiseByLabelDataset(base_test_dataset, class_noise=main_class_noise, noise_map=flip_map)
    elif config.mode == "low":
        test_dataset = AddNoiseByLabelDataset(base_test_dataset, class_noise=main_class_noise, noise_map=None)


    num_data = int(len(train_dataset) * .04)
    print(num_data)
    indices = np.random.choice(len(train_dataset), num_data, replace=False)
    train_indices = indices[:int(num_data * .5)]
    val_indices = indices[int(num_data * .5):]

    hidden_dim = 256
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(784, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 10),
            )

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.model(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = MLP().to(device)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(
        Subset(train_dataset, train_indices), batch_size=int(len(train_indices)), shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
        Subset(train_dataset, val_indices), batch_size=int(len(val_indices)), shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=int(len(test_dataset)), shuffle=False, num_workers=2)

    optimizer = build_optimizer(OptimizerSetting(name=config.optimizer,
                                                lr=config.lr,
                                                weight_decay=config.weight_decay,
                                                model=model,
                                                momentum=config.momentum,
                                                eps=config.eps,
                                                beta_1=config.beta_1,
                                                beta_2=config.beta_2,
                                                rho=config.rho,
                                                damping=config.damping,
                                                epsilon=config.epsilon,
                                                update_freq=config.update_freq,
                                                hessian_power=config.hessian_power,
                                                ))

    def closure():
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param).pow(2)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels) + config.weight_decay * l2_reg
        loss.backward()
        return loss

    step_count = 0
    for epoch in range(0, config.total_epochs):

        print(f'epoch: {epoch} start')

        avg_batch_time = 0.0
        avg_loss = 0.0

        model.train()
        
        for i, (inputs, labels) in enumerate(train_loader):
            start_time = time.time()

            inputs  = inputs.to(device)
            labels  = labels.to(device)

            if config.optimizer == "quasinewton":
                loss = optimizer.step(closure=closure)
                avg_loss += loss.item()
            elif config.optimizer == "sam":
                logits = model(inputs)
                loss = loss_fn(logits, labels)
                avg_loss += loss.item()
                
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                logits = model(inputs)
                loss_fn(logits, labels).backward()
                optimizer.second_step(zero_grad=True)
            else:
                logits = model(inputs)

                loss = loss_fn(logits, labels)
                avg_loss += loss.item()

                if config.optimizer == "adahessian" or config.optimizer == "sophiah":
                    loss.backward(create_graph=True)
                else:
                    loss.backward()
                optimizer.step()

            step_count += 1
            if config.optimizer != "sam":
                optimizer.zero_grad()
            if config.use_scheduler:
                scheduler.step()

            batch_time = time.time() - start_time
            avg_batch_time += batch_time

        avg_batch_time /= (i + 1)
        avg_loss /= (i + 1)
        
        print("Evaluating on Val")
        val_metrics = eval_single_dataset(model, val_loader, device)
        print("Evaluating on Test")
        test_metrics = eval_single_dataset(model, test_loader, device)

        print(f'epoch: {epoch} done')



if __name__ == "__main__":
    config = Config()
    trainer(config)