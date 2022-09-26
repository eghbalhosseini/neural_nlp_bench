from datasets import load_dataset
from transformers import AutoTokenizer
def tokenize(batch):
    batch["text"] = [tokenizer(examples) for examples in batch['text']]
    return batch



if __name__ =='__main__':
    dset = load_dataset("openwebtext")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", fast=False)
    dset_tokenized = dset.map(tokenize, batched=True)
    unique_tokens = dset_tokenized.unique("text_col", flatten=True)  # <- new argument
    fdist = FreqDist(unique_tokens)