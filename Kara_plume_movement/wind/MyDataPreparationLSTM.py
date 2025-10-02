import numpy as np
import pickle
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader, Sampler

import random


class CustomDataset(TorchDataset):
    def __init__(self, file_paths, num_days=14, num_years=4):
        """
        Инициализация Dataset.

        :param file_paths: список путей до pickle-файлов.
        :param num_days: длина последовательности дней.
        """
        file_paths.sort()
        self.file_paths = file_paths
        self.num_days = num_days
        self.num_years = num_years
        
        # Список всех уникальных лет, извлеченный из имен файлов
        self.years = self._extract_years()
        
        # Хранилище загруженных данных по выбранным годам
        self.data = None
        self.indices = None
    
    def _extract_years(self):
        """
        Извлекает уникальные годы из списка путей на основе шаблона имени файла.
        Предполагается, что файлы содержат год в названии, например "data/1979_01.pkl".

        :param file_paths: список путей до файлов.
        :return: множество уникальных лет.
        """
        years = []
        for path in self.file_paths:
            # Извлекаем год из названия файла с использованием предположения, что год — 4 цифры
            year = int("".join(filter(str.isdigit, path.split("/")[-1][:4])))
            years.append(year)
        return sorted(set(years))
    
    def _load_and_process_years(self, selected_years):
        """
        Загружает и соединяет данные для выбранных годов.

        :param selected_years: список выбранных годов.
        """
        all_data = []
        for file_path in self.file_paths:
            # Сравниваем год из файла с выбранными годами
            year = int("".join(filter(str.isdigit, file_path.split("/")[-1][:4])))
            if year in selected_years:
                try:
                    # Загружаем данные из pkl файла
                    with open(file_path, 'rb') as f:
                        monthly_array = pickle.load(f)

                    # Формируем данные: каждая строка - отдельный день
                    daily_data = monthly_array.reshape(-1, 24, monthly_array.shape[1], monthly_array.shape[2], monthly_array.shape[3]).mean(axis=1)
                    all_data.append(daily_data)
                except Exception as e:
                    print(f"Ошибка загрузки файла {file_path}: {e}")

        # Конкатенируем данные по всем выбранным годам
        if all_data:
            self.data = np.concatenate(all_data, axis=0)
        else:
            self.data = None

    def select_random_years(self):
        """
        Сбрасывает текущие данные и случайным образом выбирает несколько лет для загрузки.

        """
        if self.num_years > len(self.years):
            raise ValueError("Количество запрашиваемых лет превышает доступное.")

        # Выбираем случайные годы
        selected_years = random.sample(self.years, self.num_years)

        # Загружаем данные за выбранные годы
        self._load_and_process_years(selected_years)

        # Генерируем индексы последовательностей
        if self.data is not None:
            self.indices = self._generate_indices()
        else:
            self.indices = []

    def _generate_indices(self):
        """
        Генерирует индексы доступных последовательностей.
        :return: список индексов.
        """
        indices = []
        for j in range(self.num_years):
            for i in range(len(self.data) // self.num_years * j, (len(self.data) // self.num_years) * (j + 1) - self.num_days):
                indices.append([i, i + self.num_days])
        return indices

    def clear_data(self):
        """
        Очищает загруженные данные.
        """
        self.data = None
        self.indices = None

    def __len__(self):
        """
        Возвращает количество доступных последовательностей.
        """
        return len(self.indices) if self.indices else 0

    def __getitem__(self, idx):
        """
        Возвращает последовательность и целевой признак для заданного индекса.

        :param idx: индекс элемента.
        :return: tuple (sequence, target).
        """
        if self.data is None:
            raise RuntimeError("Данные не загружены. Используйте метод select_random_years().")

        start_idx, end_idx = self.indices[idx]
        sequence = self.data[start_idx:end_idx]
        target = self.data[end_idx]
        return sequence, target
