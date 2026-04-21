import datetime
from tqdm import tqdm
import os
import fnmatch
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
import torch.nn.functional as F
from torchvision import datasets, transforms
from typing import Tuple, List, Type, Dict, Any
from torch.utils.tensorboard import SummaryWriter

from SGDR import CosineAnnealingWarmRestarts
from MyDataPreparationSeq import CustomDataset
from TransformerMask1d import SequenceToVectorTransformer

device = torch.device('cuda:1')


def LogMessage(log_fname, msg):
    with open(log_fname, 'a') as logf:
        logf.write('================ ' + str(datetime.datetime.now()) + ' ================\n')
        logf.write(msg)
        logf.write('\n')


def find_files(directory, pattern, maxdepth=None):
    flist = []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                filename = filename.replace('\\\\', os.sep)
                if maxdepth is None:
                    flist.append(filename)
                else:
                    if filename.count(os.sep)-directory.count(os.sep) <= maxdepth:
                        flist.append(filename)
    return flist


def mask_one_timestep(x: torch.Tensor, mask_value: float = 0.0):
    """
    Маскирует ровно один timestep в каждой последовательности.

    Args:
        x: [B, T, D]
        mask_value: чем заменять замаскированный вектор (например 0.0)

    Returns:
        x_masked: [B, T, D]  (в одной позиции t* вектор заменён на mask_value)
        target:   [B, D]     (исходный вектор, который нужно восстановить)
        t_idx:    [B]        (индексы замаскированных timestep для каждого элемента батча)
    """
    assert x.dim() == 3, f"Expected [B,T,D], got {tuple(x.shape)}"
    B, T, D = x.shape
    device = x.device

    # Случайный индекс timestep для каждого объекта в батче
    t_idx = torch.randint(low=0, high=T, size=(B,), device=device)  # [B]

    # Target = исходный вектор в позиции t_idx
    b_idx = torch.arange(B, device=device)  # [B]
    target = x[b_idx, t_idx, :].clone()     # [B, D]

    # Делаем замаскированную копию входа
    x_masked = x.clone()
    x_masked[b_idx, t_idx, :] = mask_value

    return x_masked, target, t_idx


def train_single_epoch(model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer, 
                       loss_function: torch.nn.Module, 
                       dataloader: torch.utils.data.DataLoader,
                       batch_size: int,
                       size: int,
                       mask_value: float = 0.0):

    model.train()
    train_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()
        x = batch.to(device='cuda', dtype=torch.float)

        # Маскирование через mask_one_timestep
        x_masked, target, t_idx = mask_one_timestep(x, mask_value=mask_value)

        # Модель возвращает предсказание замаскированного вектора
        pred = model(x_masked)  # [B, D]

        # Loss
        loss = loss_function(pred, target)

        loss.backward()
        optimizer.step()

        train_loss += loss.detach() * batch_size

    return {'loss': train_loss.item() / size}


def validate_single_epoch(model: torch.nn.Module,
                          loss_function: torch.nn.Module,
                          dataloader: torch.utils.data.DataLoader,
                          batch_size: int,
                          size: int,
                          mask_value: float = 0.0):
    
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for batch_data in dataloader:
            x = batch_data.to(device='cuda', dtype=torch.float)  # [N, T, D]

            # Маскирование через mask_one_timestep
            x_masked, target, t_idx = mask_one_timestep(x, mask_value=mask_value)

            # Forward
            pred = model.forward(x_masked)  # [N, D]

            # Loss
            loss = loss_function(pred, target)
            test_loss += loss.detach() * batch_size
            
    return {'loss': test_loss.item() / size}


def train_model(run_name: str,
                model: torch.nn.Module,
                dataset: torch.utils.data.Dataset,
                dataloader: torch.utils.data.DataLoader,
                loss_function: torch.nn.Module,
                optimizer_class: Type[torch.optim.Optimizer] = torch.optim,
                optimizer_params: Dict = {},
                initial_lr = 0.001,
                lr_scheduler_class: Any = torch.optim.lr_scheduler.ReduceLROnPlateau,
                lr_scheduler_params: Dict = {},
                max_epochs = 1000,
                early_stopping_patience = 10):
    
    tb_writer = SummaryWriter(log_dir=f'/app/Kara_plume_movement/wind/logs/{run_name}/')
    
    params_to_optimize = [
        {'params': model.parameters()},
    ]
    
    LogMessage(f'/app/Kara_plume_movement/wind/descriptions/{run_name}_description.txt', 'start')
    
    optimizer = torch.optim.Adam(params_to_optimize, lr=initial_lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=64, T_mult=2, eta_min=1.0e-9, lr_decay=0.75)
    
    loss_history_train = []
    loss_history_test = []
    
    batch_size = 16
    pbar = tqdm(total=max_epochs)
    for epoch in range(max_epochs):
        dataset.clear_cache()
        train_loss = train_single_epoch(model, optimizer, loss_function, dataloader, batch_size, len(dataset))

        dataset.clear_cache()
        test_loss = validate_single_epoch(model, loss_function, dataloader, batch_size, len(dataset))
        
        loss_history_train.append(train_loss['loss'])
        loss_history_test.append(test_loss['loss'])
        
        tb_writer.add_scalar('train_loss', train_loss['loss'], global_step=epoch)
        tb_writer.add_scalar('test_loss', test_loss['loss'], global_step=epoch)
        tb_writer.add_scalar('lr', scheduler.get_last_lr()[-1], global_step=epoch)
        
        LogMessage(f'/app/Kara_plume_movement/wind/descriptions/{run_name}_description.txt',
                   f'epoch = {epoch}, train_loss = {train_loss["loss"]}, test_loss = {test_loss["loss"]}')
        
        scheduler.step(epoch=epoch)
        
        pbar.update(1)
    
    torch.save(model, f'/app/Kara_plume_movement/wind/models/model_{run_name}.pth')


if __name__ == '__main__':
    dataset = CustomDataset(data_path="/app/Kara_plume_movement/extracted_features/extracted_wind_007_6hrs", years=[year for year in range(1979, 2024+1)])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    model = SequenceToVectorTransformer(
        in_features=64,
        d_model=128,
        nhead=4,
        num_layers=3,
        out_dim=64,
        mask_p=0.2,
        mask_value=0.0,
        pooling="mean",  # или "cls"
    )

    model = model.cuda()
    run_name = 'wind_Transformer1D_run002'

    train_model(run_name,
                model=model,
                dataset=dataset,
                dataloader=dataloader,
                loss_function=torch.nn.MSELoss(),
                initial_lr=0.0001,
                max_epochs=960)
