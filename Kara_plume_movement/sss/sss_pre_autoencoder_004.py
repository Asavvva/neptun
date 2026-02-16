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
from MyDataPreparationMix_norm import CustomDataset
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


def train_single_epoch(encoder: torch.nn.Module,
                       decoder: torch.nn.Module,
                       optimizer: torch.optim.Optimizer, 
                       loss_function: torch.nn.Module, 
                       dataset: torch.utils.data.Dataset,
                       dataloader: torch.utils.data.DataLoader,
                       batch_size: int):
    
    encoder.train()
    decoder.train()
    train_loss = 0
    
    size = len(dataset)
    
    for batch_data in dataloader:
        optimizer.zero_grad()
        data, mask = batch_data
        data_gpu = data.to(device='cuda', dtype=torch.float)
        mask_gpu = mask.to(device='cuda', dtype=torch.float)

        encoded_data = encoder.forward(data_gpu)
        decoded_data = decoder.forward(encoded_data)
        
        data_gpu_masked = data_gpu[mask_gpu == 1]
        result_masked = decoded_data[mask_gpu == 1]

        loss = loss_function(data_gpu_masked, result_masked)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.detach() * batch_size
        
    return {'loss': train_loss.item() / size}


def validate_single_epoch(encoder: torch.nn.Module,
                          decoder: torch.nn.Module,
                          loss_function: torch.nn.Module,
                          dataset: torch.utils.data.Dataset,
                          dataloader: torch.utils.data.DataLoader,
                          batch_size: int):
    
    encoder.eval()
    decoder.eval()
    test_loss = 0
    
    size = len(dataset)
    
    with torch.no_grad():
        for batch_data in dataloader:
            data, mask = batch_data
            data_gpu = data.to(device='cuda', dtype=torch.float)
            mask_gpu = mask.to(device='cuda', dtype=torch.float)
            
            encoded_data = encoder.forward(data_gpu)
            decoded_data = decoder.forward(encoded_data)
            
            data_gpu_masked = data_gpu[mask_gpu == 1]
            result_masked = decoded_data[mask_gpu == 1]
            
            loss = loss_function(data_gpu_masked, result_masked)
            test_loss += loss.detach() * batch_size
            
    return {'loss': test_loss.item() / size}


def train_model(run_name: str,
                encoder: torch.nn.Module,
                decoder: torch.nn.Module,
                dataset: torch.utils.data.Dataset,
                dataloader: torch.utils.data.DataLoader,
                batch_size: int,
                loss_function: torch.nn.Module,
                optimizer_class: Type[torch.optim.Optimizer] = torch.optim,
                optimizer_params: Dict = {},
                initial_lr = 0.001,
                lr_scheduler_class: Any = torch.optim.lr_scheduler.ReduceLROnPlateau,
                lr_scheduler_params: Dict = {},
                max_epochs = 1000,
                early_stopping_patience = 10):
    
    tb_writer = SummaryWriter(log_dir=f'/app/Kara_plume_movement/sss/logs/{run_name}/')
    
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()},
    ]
    
    LogMessage(f'/app/Kara_plume_movement/sss/descriptions/{run_name}_description.txt', 'start')
    
    optimizer = torch.optim.Adam(params_to_optimize, lr=initial_lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=64, T_mult=2, eta_min=1.0e-9, lr_decay=0.75)
    
    loss_history_train = []
    loss_history_test = []
    
    batch_size = batch_size
    pbar = tqdm(total=max_epochs)
    for epoch in range(max_epochs):
        train_loss = train_single_epoch(encoder, decoder, optimizer, loss_function, dataset, dataloader, batch_size)
        
        test_loss = validate_single_epoch(encoder, decoder, loss_function, dataset, dataloader, batch_size)
        
        loss_history_train.append(train_loss['loss'])
        loss_history_test.append(test_loss['loss'])
        
        tb_writer.add_scalar('train_loss', train_loss['loss'], global_step=epoch)
        tb_writer.add_scalar('test_loss', test_loss['loss'], global_step=epoch)
        tb_writer.add_scalar('lr', scheduler.get_last_lr()[-1], global_step=epoch)
        
        LogMessage(f'/app/Kara_plume_movement/sss/descriptions/{run_name}_description.txt',
                   f'epoch = {epoch}, train_loss = {train_loss["loss"]}, test_loss = {test_loss["loss"]}')
        
        scheduler.step(epoch=epoch)
        
        pbar.update(1)
    
    torch.save(encoder, f'/app/Kara_plume_movement/sss/models/model_{run_name}_encoder.pth')
    torch.save(decoder, f'/app/Kara_plume_movement/sss/models/model_{run_name}_decoder.pth')


if __name__ == '__main__':
    batch_size = 32

    dates1 = {
        '2010': {'start': '0711', 'finish': '1009'},
        '2011': {'start': '0625', 'finish': '1029'},
        '2012': {'start': '0701', 'finish': '1105'},
        '2013': {'start': '0713', 'finish': '1018'},
        '2014': {'start': '0716', 'finish': '1023'},
        '2015': {'start': '0630', 'finish': '1025'},
        '2016': {'start': '0709', 'finish': '1031'},
        '2017': {'start': '0715', 'finish': '1020'},
        '2018': {'start': '0801', 'finish': '1031'},
        '2019': {'start': '0710', 'finish': '1025'},
        '2020': {'start': '0707', 'finish': '1031'},
        '2021': {'start': '0710', 'finish': '1025'},
        '2022': {'start': '0701', 'finish': '1031'},
        '2023': {'start': '0720', 'finish': '1010'},
    }

    dates2 = {
        '2015': {'start': '0630', 'finish': '1025'},
        '2016': {'start': '0709', 'finish': '1031'},
        '2017': {'start': '0715', 'finish': '1020'},
        '2018': {'start': '0801', 'finish': '1031'},
        '2019': {'start': '0710', 'finish': '1025'},
        '2020': {'start': '0707', 'finish': '1031'},
        '2021': {'start': '0710', 'finish': '1025'},
        '2022': {'start': '0701', 'finish': '1031'},
        '2023': {'start': '0720', 'finish': '1010'},
    }

    dataset = CustomDataset(dates_dict1=dates1, path1='/mnt/hippocamp/asavin/data/ESACCI/ESACCI_norm',
                            dates_dict2=dates2, path2='/mnt/hippocamp/asavin/data/SSS_ESACCI_grid/SSS_ESACCI_grid_norm_free_ice_dates')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    encoder = Encoder(in_channels=1, H=74, W=52, expansions=[4, 4, 4, 4, 4], n_blocks=50, decreases=[2, 2, 2, 2, 2], bottleneck=64)
    decoder = Decoder(in_features=encoder.bottleneck, start_channels=1024, finish_channels=encoder.in_channels, n_layers=5,
                      expansion_value=0.25, increase_value=2, H=3, W=2, H_out=74, W_out=52)

    encoder = encoder.cuda()
    decoder = decoder.cuda()

    run_name = 'sss_pre_autoencoder_run004'

    train_model(run_name, encoder, decoder,
                dataset=dataset,
                dataloader=dataloader,
                batch_size=batch_size,
                loss_function=torch.nn.MSELoss(),
                initial_lr=0.0001,
                max_epochs=960)
