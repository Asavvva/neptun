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
from MyDataPreparation import CustomDataset, Sampler
from autoencoder import Encoder, Decoder

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


def make_wind_products(zu, zv, zr, file):
    '''
    Из предсказанных характеристик восстанавливаются
    статистики ветра для дополнительных лоссов
    '''
    u = zu * file['u10_std'] + file['u10_mean']
    v = zv * file['v10_std'] + file['v10_mean']

    tr = zr * file['tr_std'] + file['tr_mean']
    r = file['r_mean'] * torch.expm1(tr)
    tr2_pred = torch.log1p(r**2 / file['r2_5p'])
    zr2_pred = (tr2_pred - file['tr2_mean']) / file['tr2_std']

    tru_pred = torch.asinh((r * u) / file['ru_median'])
    zru_pred = (tru_pred - file['tru_mean']) / file['tru_std']

    trv_pred = torch.asinh((r * v) / file['rv_median'])
    zrv_pred = (trv_pred - file['trv_mean']) / file['trv_std']

    return zr2_pred, zru_pred, zrv_pred


def train_single_epoch(encoder: torch.nn.Module,
                       decoder: torch.nn.Module,
                       optimizer: torch.optim.Optimizer, 
                       loss_function: torch.nn.Module, 
                       dataset: torch.utils.data.Dataset,
                       batch_size: int,
                       file: dict,
                       lambda_u: float, lambda_v: float, lambda_r: float,
                       lambda_r2: float, lambda_ru: float, lambda_rv: float):
    
    encoder.train()
    decoder.train()
    train_loss = 0
    
    dataset.make_new_data()
    sampler = Sampler([i for i in range(dataset.wind_array.shape[0])], shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, sampler=sampler)
    
    size = len(dataset)
    
    for batch_data in dataloader:
        optimizer.zero_grad()
        wind_gpu = batch_data.to(device='cuda', dtype=torch.float)

        zu, zv, zr, zr2, zru, zrv = wind_gpu.unbind(dim=1)
        learning_data = torch.stack([zu, zv, zr], axis=1)
        
        encoded_data = encoder.forward(learning_data)
        decoded_data = decoder.forward(encoded_data)

        zu_pred, zv_pred, zr_pred = decoded_data.unbind(dim=1)
        zr2_pred, zru_pred, zrv_pred = make_wind_products(zu_pred, zv_pred, zr_pred, file)

        loss_u = lambda_u * loss_function(zu, zu_pred)
        loss_v = lambda_v * loss_function(zv, zv_pred)
        loss_r = lambda_r * loss_function(zr, zr_pred)

        loss_r2 = lambda_r2 * loss_function(zr2, zr2_pred)
        loss_ru = lambda_ru * loss_function(zru, zru_pred)
        loss_rv = lambda_rv * loss_function(zrv, zrv_pred)
        
        loss = loss_u + loss_v + loss_r + loss_r2 + loss_ru + loss_rv
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.detach() * batch_size
        
    dataset.clear_cache()
    return {'loss': train_loss.item() / size, 'loss_u': loss_u.item() / size, 'loss_v': loss_v.item() / size, 'loss_r': loss_r.item() / size,
            'loss_r2': loss_r2.item() / size, 'loss_ru': loss_ru.item() / size, 'loss_rv': loss_rv.item() / size}


def validate_single_epoch(encoder: torch.nn.Module,
                          decoder: torch.nn.Module,
                          loss_function: torch.nn.Module,
                          dataset: torch.utils.data.Dataset,
                          batch_size: int,
                          file: dict,
                          lambda_u: float, lambda_v: float, lambda_r: float,
                          lambda_r2: float, lambda_ru: float, lambda_rv: float):
    
    encoder.eval()
    decoder.eval()
    test_loss = 0
    
    dataset.make_new_data()
    sampler = Sampler([i for i in range(dataset.wind_array.shape[0])], shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, sampler=sampler)
    
    size = len(dataset)
    
    with torch.no_grad():
        for batch_data in dataloader:
            wind_gpu = batch_data.to(device='cuda', dtype=torch.float)

            zu, zv, zr, zr2, zru, zrv = wind_gpu.unbind(dim=1)
            learning_data = torch.stack([zu, zv, zr], axis=1)
            
            encoded_data = encoder.forward(learning_data)
            decoded_data = decoder.forward(encoded_data)

            zu_pred, zv_pred, zr_pred = decoded_data.unbind(dim=1)
            zr2_pred, zru_pred, zrv_pred = make_wind_products(zu_pred, zv_pred, zr_pred, file)

            loss_u = lambda_u * loss_function(zu, zu_pred)
            loss_v = lambda_v * loss_function(zv, zv_pred)
            loss_r = lambda_r * loss_function(zr, zr_pred)

            loss_r2 = lambda_r2 * loss_function(zr2, zr2_pred)
            loss_ru = lambda_ru * loss_function(zru, zru_pred)
            loss_rv = lambda_rv * loss_function(zrv, zrv_pred)
            
            loss = loss_u + loss_v + loss_r + loss_r2 + loss_ru + loss_rv

            test_loss += loss.detach() * batch_size
            
    dataset.clear_cache()
    return {'loss': test_loss.item() / size, 'loss_u': loss_u.item() / size, 'loss_v': loss_v.item() / size, 'loss_r': loss_r.item() / size,
            'loss_r2': loss_r2.item() / size, 'loss_ru': loss_ru.item() / size, 'loss_rv': loss_rv.item() / size}


def train_model(run_name: str,
                encoder: torch.nn.Module,
                decoder: torch.nn.Module,
                dataset: torch.utils.data.Dataset,
                file: dict,
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
        {'params': encoder.parameters()},
        {'params': decoder.parameters()},
    ]
    
    LogMessage(f'/app/Kara_plume_movement/wind/descriptions/{run_name}_description.txt', 'start')
    
    optimizer = torch.optim.Adam(params_to_optimize, lr=initial_lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=64, T_mult=2, eta_min=1.0e-9, lr_decay=0.75)

    lambda_u, lambda_v, lambda_r = 1.0, 1.0, 1.0
    lambda_r2, lambda_ru, lambda_rv = 0.2, 0.2, 0.2
    
    loss_history = {
        'loss_history_train': [],
        'loss_u_history_train': [],
        'loss_v_history_train': [],
        'loss_r_history_train': [],
        'loss_r2_history_train': [],
        'loss_ru_history_train': [],
        'loss_rv_history_train': [],
        'loss_history_test': [],
        'loss_u_history_test': [],
        'loss_v_history_test': [],
        'loss_r_history_test': [],
        'loss_r2_history_test': [],
        'loss_ru_history_test': [],
        'loss_rv_history_test': [],
    }
    
    batch_size = 16
    pbar = tqdm(total=max_epochs)
    for epoch in range(max_epochs):
        train_loss = train_single_epoch(encoder, decoder, optimizer, loss_function, dataset, batch_size, file, 
                                        lambda_u, lambda_v, lambda_r, lambda_r2, lambda_ru, lambda_rv)
        
        test_loss = validate_single_epoch(encoder, decoder, loss_function, dataset, batch_size, file,
                                          lambda_u, lambda_v, lambda_r, lambda_r2, lambda_ru, lambda_rv)
        
        for key in train_loss:
            loss_history[f'{key}_history_train'].append(train_loss[f'{key}'])

        for key in test_loss:
            loss_history[f'{key}_history_test'].append(test_loss[f'{key}'])
            
        tb_writer.add_scalar('lr', scheduler.get_last_lr()[-1], global_step=epoch)

        for key in train_loss:
            tb_writer.add_scalar(f'train_{key}', train_loss[f'{key}'], global_step=epoch)

        for key in test_loss:
            tb_writer.add_scalar(f'test_{key}', test_loss[f'{key}'], global_step=epoch)
        
        LogMessage(f'/app/Kara_plume_movement/wind/descriptions/{run_name}_description.txt',
                   f'epoch = {epoch}, train_loss = {train_loss["loss"]}, test_loss = {test_loss["loss"]}')
        
        scheduler.step(epoch=epoch)
        
        pbar.update(1)
    
    torch.save(encoder, f'/app/Kara_plume_movement/wind/models/model_{run_name}_encoder.pth')
    torch.save(decoder, f'/app/Kara_plume_movement/wind/models/model_{run_name}_decoder.pth')


if __name__ == '__main__':
    with open(f'/mnt/hippocamp/asavin/data/wind/wind_products_data/wind_products_norm_params.pkl', 'rb') as file:
        wind_products_norm_params = pickle.load(file)

    wind_files_pkl = find_files('/mnt/hippocamp/asavin/data/wind/wind_products_arrays_kara_norm_n80_s70_w55_e105', '*.pkl')
    wind_files_pkl.sort()

    dataset = CustomDataset(wind_files_pkl, n_files=30)

    encoder = Encoder(in_channels=3, H=41, W=201, expansions=[4, 4, 4, 4], n_blocks=32, decreases=[2, 2, 2, 2], bottleneck=64)
    decoder = Decoder(in_features=encoder.bottleneck, start_channels=768, finish_channels=encoder.in_channels, n_layers=4,
                      expansion_value=0.25, increase_value=2, H=3, W=13, H_out=41, W_out=201)

    encoder = encoder.cuda()
    decoder = decoder.cuda()

    run_name = 'wind_pre_autoencoder_run005'

    train_model(run_name, encoder, decoder,
                dataset=dataset,
                file = wind_products_norm_params,
                loss_function=torch.nn.MSELoss(),
                initial_lr=0.0001,
                max_epochs=448)
