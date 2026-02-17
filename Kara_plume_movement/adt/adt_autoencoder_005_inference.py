from netCDF4 import Dataset
import datetime
from tqdm import tqdm
import os
import fnmatch
import pickle

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset as TorchDataset
import torch.nn.functional as F
from torchvision import datasets, transforms
from typing import Tuple, List, Type, Dict, Any
from torch.utils.tensorboard import SummaryWriter

from SGDR import CosineAnnealingWarmRestarts
from MyDataPreparation_0125_norm import CustomDataset, Sampler
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


def inference_single_file(encoder: torch.nn.Module,
                          dataloader: torch.utils.data.DataLoader):
    
    encoder.eval()
    outputs_list = []

    with torch.no_grad():
        for batch_data in dataloader:
            data, _, _ = batch_data
            data_gpu = data.to(device='cuda', dtype=torch.float)

            encoded_data = encoder.forward(data_gpu)
            outputs_list.append(encoded_data.detach().cpu())
            
    outputs = torch.cat(outputs_list, dim=0)
    return outputs


def inference_model(run_name: str,
                    encoder: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader):
    
    outputs = inference_single_file(encoder, dataloader)
    with open(f'/app/Kara_plume_movement/extracted_features/extracted_adt_005/encoded_adt_1993-2024_daily_n80_s70_w55_e105.pkl', 'wb') as f:
        pickle.dump(outputs.numpy(), f)


if __name__ == '__main__':
    data = Dataset('/mnt/hippocamp/asavin/data/adt/adt_1993-2024_daily_n80_s70_w55_e105.nc', 'r')

    batch_size = 32
    dataset = CustomDataset(data=data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data.close()

    run_name = 'adt_pre_autoencoder_run005'
    encoder = torch.load(f'/app/Kara_plume_movement/adt/models/model_{run_name}_encoder.pth', map_location=torch.device('cpu'));
    encoder = encoder.cuda()

    inference_model(run_name, encoder, dataloader)
