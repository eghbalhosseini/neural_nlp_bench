from tqdm import tqdm
from transformers import AutoTokenizer
from accelerate import accelerator
from datasets import load_dataset
import numpy as np
def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples['text'], return_length=True)
    return {"length": outputs['length']}

if __name__ == "__main__":
    open_web = load_dataset('openwebtext', split='train')
    tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
    #small = open_web.select(list(np.arange(2000)))
    tokenized_datasets = open_web.map(tokenize_function, batched=True, remove_columns=["text"],batch_size=1000)
    num_tokens=np.sum(tokenized_datasets['length'])
    #test = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    print(f"--------------{num_tokens=}-------------")


