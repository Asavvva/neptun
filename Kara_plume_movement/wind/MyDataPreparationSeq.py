import torch
from torch.utils.data import Dataset
import os
import pickle
import glob
import random
import numpy as np
from datetime import datetime

class CustomDataset(Dataset):
    def __init__(self, data_path, years, seq_length=4*7, months=list(range(6, 12)), transform=None):
        """
        data_path: путь к директории с файлами *.pkl
        years: список годов для обучения (например, [2018, 2019, 2020])
        seq_length: длина последовательности (по умолчанию 4*7, неделя)
        months: месяцы, которые хотим брать (по умолчанию с июня по ноябрь)
        transform: опциональная трансформация для данных
        """
        self.data_path = data_path
        self.years = years
        self.seq_length = seq_length
        self.months = months
        self.transform = transform

        self.year = None
        self.data = None
        self.seq_indices = None

    def get_new_data(self):
        self._reload_year()
    
    def _reload_year(self):
        self.year = random.choice(self.years)
        self.data = []
        self.month_lens = []
        for month in self.months:
            fname = f"{self.year}-{month:02d}.pkl"
            fpath = os.path.join(self.data_path, fname)
            with open(fpath, 'rb') as f:
                data_part = pickle.load(f)
            self.data.append(data_part)
            self.month_lens.append(data_part.shape[0])
        self.data = np.concatenate(self.data, axis=0)
        total_len = self.data.shape[0]

        # Индексы от 0 до total_len - seq_length включительно
        # Все последовательности start:start+seq_length точно лежат внутри июня-ноября
        self.seq_indices = [i for i in range(0, total_len - self.seq_length + 1)]
        random.shuffle(self.seq_indices)

    def __len__(self):
        return len(self.seq_indices)

    def __getitem__(self, idx):
        start = self.seq_indices[idx]
        end = start + self.seq_length
        seq = self.data[start:end]
        if self.transform is not None:
            seq = self.transform(seq)
        return torch.tensor(seq, dtype=torch.float)

    def clear_cache(self):
        # Очищаем кэш данных
        self.data = None
        self.seq_indices = None


# dataset = CustomDataset(data_path="/path/to/pkls", years=[2018, 2019, 2020])
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
# dataset.get_new_data()
# for epoch in range(num_epochs):
#     for batch in dataloader:
#         ... # обучение
#     dataset.clear_cache()
