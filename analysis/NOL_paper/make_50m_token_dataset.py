import os
import torch
from torch.utils.data import DataLoader
import datasets
from datasets import load_dataset, load_from_disk

if __name__ == "__main__":
    data = '1B'
    mini_dataset = load_dataset('/om/user/ehoseini/MyData/miniBERTa_v2/', f'miniBERTa-{data}')
