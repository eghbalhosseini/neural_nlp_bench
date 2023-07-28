import os
import torch
from torch.utils.data import DataLoader
import datasets
from datasets import load_dataset, load_from_disk
import numpy as np
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import jsonlines

if __name__ == "__main__":
    data = '1B'
    mini_dataset = load_dataset('/Users/eghbalhosseini/MyData/miniBERTa_v2', f'miniBERTa-{data}')

    mb_len = len(mini_dataset['train'])
    mb_sample_rel_num = ((0.05 * 0.8 / 2), (0.05 * 0.67 / 2))
    subselect_len_0 = int((mb_len * mb_sample_rel_num[0])*1.39)
    subselect_len_1 = int((mb_len * mb_sample_rel_num[1])*1.39)
    subselect_idx = np.concatenate(
        [np.arange(subselect_len_0),
         np.arange(mb_len - subselect_len_1, mb_len)])
    dataset_50m = mini_dataset['train'].select(subselect_idx)
    dataset_50m_full=mini_dataset
    dataset_50m_full['train']=dataset_50m
    dataset_50m_full.save_to_disk(os.path.join('/Users/eghbalhosseini/MyData/miniBERTa_v2', f'miniBERTa-50M'))
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # tokneize the dataset and count the number of words and tokens
    dataset_50m = load_from_disk(os.path.join('/Users/eghbalhosseini/MyData/miniBERTa_v2', f'miniBERTa-50M'))
    vers='50M'
    miniBERTa_dir='/Users/eghbalhosseini/MyData/miniBERTa_v2/'
    data=dataset_50m_full
    for key in data.keys():
        if not os.path.exists(os.path.join(miniBERTa_dir, vers, vers + '_' + key + '.jsonl.zst')):
            if not os.path.exists(os.path.join(miniBERTa_dir, vers, vers + '_' + key + '.jsonl')):
                with jsonlines.open(os.path.join(miniBERTa_dir, vers, vers + '_' + key + '.jsonl'), 'w') as writer:
                    for idy, x in tqdm(enumerate(data[key])):
                        if len(x['text']) > 0:
                            writer.write(x)
                    writer.close()

    dataset_50m = dataset_50m.map(lambda x: tokenizer(x['text']), batched=True)
    # count the number of words and tokens
    num_words_50m = 0
    num_tokens_50m = 0
    for idx, x in tqdm(enumerate(dataset_50m),total=len(dataset_50m)):
        num_tokens_50m += len(x['input_ids'])
        num_words_50m += len(x['text'].split(' '))


    dataset_10m = load_dataset('/om/user/ehoseini/MyData/miniBERTa_v2', f'miniBERTa-10M')
    dataset_10m = dataset_10m.map(lambda x: tokenizer(x['text']), batched=True)

    num_words_10m = 0
    num_tokens_10m = 0
    for idx, x in tqdm(enumerate(dataset_10m['train']),total=len(dataset_10m['train'])):
        num_words_10m += len(x['input_ids'])
        num_tokens_10m += len(x['text'].split(' '))