import datetime
from tqdm import tqdm
import os
import fnmatch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
import torch.nn.functional as F
from torchvision import datasets, transforms
from typing import Tuple, List, Type, Dict, Any
from torch.utils.tensorboard import SummaryWriter

from SGDR import CosineAnnealingWarmRestarts
from MyDataPreparationLSTM import CustomDataset
from MLP import MultiLayerPerceptron


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


def train_single_epoch(pretrained_encoder: torch.nn.Module,
                       lstm_network: torch.nn.Module,
                       lstm_decoder: torch.nn.Module,
                       pretrained_decoder: torch.nn.Module,
                       optimizer: torch.optim.Optimizer, 
                       loss_function: torch.nn.Module, 
                       dataset: torch.utils.data.Dataset,
                       batch_size: int):
    
    pretrained_encoder.eval()
    lstm_network.train()
    lstm_decoder.train()
    pretrained_decoder.eval()
    train_loss = 0

    dataset.select_random_years()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    size = len(dataset)

    for (batch_data, _) in dataloader:
        optimizer.zero_grad()
        data_gpu = batch_data.to(device='cuda', dtype=torch.float)

        data_list = data_gpu.unbind(dim=1)
        encoded_data_list = [pretrained_encoder.forward(t) for t in data_list]
        decoded_target_list = [pretrained_decoder.forward(t) for t in encoded_data_list]

        wind_vector = torch.stack(encoded_data_list, dim=1)

        encoded_data, (_, _) = lstm_network.forward(wind_vector)
        split_tensors = encoded_data.unbind(dim=1)
        processed_tensors = [pretrained_decoder.forward(lstm_decoder.forward(t)) for t in split_tensors]
        decoded_data = torch.stack(processed_tensors, dim=1)
        
        decoded_target = torch.stack(decoded_target_list, dim=1)

        loss = loss_function(decoded_target, decoded_data)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.detach() * batch_size

    dataset.clear_data()
    return {'loss': train_loss.item() / size}


def validate_single_epoch(pretrained_encoder: torch.nn.Module,
                          lstm_network: torch.nn.Module,
                          lstm_decoder: torch.nn.Module,
                          pretrained_decoder: torch.nn.Module,
                          loss_function: torch.nn.Module,
                          dataset: torch.utils.data.Dataset,
                          batch_size: int):
    
    pretrained_encoder.eval()
    lstm_network.eval()
    lstm_decoder.eval()
    pretrained_decoder.eval()
    test_loss = 0

    dataset.select_random_years()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    size = len(dataset)

    with torch.no_grad():
        for (batch_data, _) in dataloader:
            data_gpu = batch_data.to(device='cuda', dtype=torch.float)

            data_list = data_gpu.unbind(dim=1)
            encoded_data_list = [pretrained_encoder.forward(t) for t in data_list]
            decoded_target_list = [pretrained_decoder.forward(t) for t in encoded_data_list]

            wind_vector = torch.stack(encoded_data_list, dim=1)

            encoded_data, (_, _) = lstm_network.forward(wind_vector)
            split_tensors = encoded_data.unbind(dim=1)
            processed_tensors = [pretrained_decoder.forward(lstm_decoder.forward(t)) for t in split_tensors]
            decoded_data = torch.stack(processed_tensors, dim=1)
            
            decoded_target = torch.stack(decoded_target_list, dim=1)

            loss = loss_function(decoded_target, decoded_data)
            test_loss += loss.detach() * batch_size

    dataset.clear_data()
    return {'loss': test_loss.item() / size}


def train_model(run_name: str,
                pretrained_encoder: torch.nn.Module,
                lstm_network: torch.nn.Module,
                lstm_decoder: torch.nn.Module,
                pretrained_decoder: torch.nn.Module,
                dataset: torch.utils.data.Dataset,
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
        {'params': lstm_network.parameters()},
        {'params': lstm_decoder.parameters()},
    ]
    
    LogMessage(f'/app/Kara_plume_movement/wind/descriptions/{run_name}_description.txt', 'start')
    
    optimizer = torch.optim.Adam(params_to_optimize, lr=initial_lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=64, T_mult=2, eta_min=1.0e-9, lr_decay=0.75)
    
    loss_history_train = []
    loss_history_test = []
    
    batch_size = 32
    pbar = tqdm(total=max_epochs)
    for epoch in range(max_epochs):
        train_loss = train_single_epoch(pretrained_encoder, lstm_network, lstm_decoder, pretrained_decoder, optimizer, loss_function, dataset, batch_size=batch_size)
        
        test_loss = validate_single_epoch(pretrained_encoder, lstm_network, lstm_decoder, pretrained_decoder, loss_function, dataset, batch_size=batch_size)
        
        loss_history_train.append(train_loss['loss'])
        loss_history_test.append(test_loss['loss'])
        
        tb_writer.add_scalar('train_loss', train_loss['loss'], global_step=epoch)
        tb_writer.add_scalar('test_loss', test_loss['loss'], global_step=epoch)
        tb_writer.add_scalar('lr', scheduler.get_last_lr()[-1], global_step=epoch)
        
        LogMessage(f'/app/Kara_plume_movement/wind/descriptions/{run_name}_description.txt',
                   f'epoch = {epoch}, train_loss = {train_loss["loss"]}, test_loss = {test_loss["loss"]}')
        
        scheduler.step(epoch=epoch)
        
        pbar.update(1)
    
    torch.save(lstm_network, f'/app/Kara_plume_movement/wind/models/model_{run_name}_lstm_network.pth')
    torch.save(lstm_decoder, f'/app/Kara_plume_movement/wind/models/model_{run_name}_MLPdecoder.pth')


if __name__ == '__main__':
    wind_files_pkl = find_files('/mnt/hippocamp/asavin/data/wind/wind_arrays_kara_norm', '*.pkl')
    wind_files_pkl.sort()

    dataset = CustomDataset(wind_files_pkl, num_days=14, num_years=4)

    pretrained_autoencoder_name = 'wind_pre_autoencoder_run002'
    
    pretrained_encoder = torch.load(f'/app/Kara_plume_movement/wind/models/model_{pretrained_autoencoder_name}_encoder.pth', map_location=torch.device('cpu'))
    lstm_network = nn.LSTM(pretrained_encoder.bottleneck, pretrained_encoder.bottleneck, num_layers=4, batch_first=True)
    lstm_decoder = MultiLayerPerceptron(input_size=pretrained_encoder.bottleneck,
                                        output_size=pretrained_encoder.bottleneck,
                                        hidden_layers=[1024])
    pretrained_decoder = torch.load(f'/app/Kara_plume_movement/wind/models/model_{pretrained_autoencoder_name}_decoder.pth', map_location=torch.device('cpu'))

    pretrained_encoder = pretrained_encoder.cuda()
    lstm_network = lstm_network.cuda()
    lstm_decoder = lstm_decoder.cuda()
    pretrained_decoder = pretrained_decoder.cuda()

    run_name = 'wind_pre_lstm2D_run004'

    train_model(run_name, pretrained_encoder, lstm_network, lstm_decoder, pretrained_decoder,
                dataset=dataset,
                loss_function=torch.nn.MSELoss(),
                initial_lr=0.0001,
                max_epochs=960)
