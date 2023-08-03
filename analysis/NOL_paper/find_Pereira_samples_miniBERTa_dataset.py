import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
import datasets
from datasets import load_dataset, load_from_disk
import numpy as np
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import jsonlines
import pandas as pd

import spacy
from pysbd.utils import PySBDFactory

nlp = spacy.blank('en')

# explicitly adding component to pipeline
# (recommended - makes it more readable to tell what's going on)
nlp.add_pipe(PySBDFactory(nlp))
import pysbd


import mmap
import multiprocessing

def search_in_chunk(chunk, pattern, chunk_start_line):
    found_lines = []
    start = 0
    line_number = chunk_start_line
    while True:
        end = chunk.find(b'\n', start)
        if end == -1:
            break
        line = chunk[start:end].decode('utf-8')
        if pattern in line:
            found_lines.append((line_number, line.strip()))
        line_number += 1
        start = end + 1
    return found_lines

def search_pattern_in_large_file(file_path, pattern, num_processes=8):
    found_lines = []
    with open(file_path, 'rb') as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
            file_size = mmapped_file.size()
            chunk_size = file_size // num_processes
            pool = multiprocessing.Pool(processes=num_processes)
            results = []
            for i in range(num_processes):
                start = i * chunk_size
                end = start + chunk_size if i < num_processes - 1 else file_size
                chunk = mmapped_file[start:end]
                results.append(
                    pool.apply_async(search_in_chunk, (chunk, pattern))
                )
            pool.close()
            pool.join()
            for result in results:
                found_lines.extend(result.get())
    return found_lines

if __name__ == "__main__":
    data = '1B'
    file_path = f'/Users/eghbalhosseini/MyData/miniBERTa_v2/miniBERTa-{data}/proc_data_train_for_{data}.raw'

    Pereira_stim=pd.read_csv('/Users/eghbalhosseini/.neural_nlp/Pereira2018-stimulus_set.csv')
    pereira_sentences=Pereira_stim['sentence'].tolist()

    sentence_exist={}
    for idx, per_sent in tqdm(enumerate(pereira_sentences),total=len(pereira_sentences)):
        pattern_to_search=per_sent
        found_lines = search_pattern_in_large_file(file_path, pattern_to_search)
        sentence_instnace=[]
        if found_lines:
            for line_number, line in found_lines:
                print(f'Line {line_number}: {line}')
                sentence_instnace.append(per_sent)
        sentence_exist[per_sent]=sentence_instnace

    np.sum([len(sentence_exist[key]) for key in sentence_exist.keys()])


    dataset_train = mini_dataset['train']
    seg=pysbd.Segmenter(language="en", clean=False)
    sentence_list=[]
    for id, x in tqdm(enumerate(dataset_train),total=len(dataset_train)):
        sentences=seg.segment(x['text'])
        for s in sentences:
            sentence_list.append(s)
