from neural_nlp import model_pool
from neural_nlp.models.implementations import transformer_configurations
import transformers
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoConfig, AutoModel, AutoModelWithLMHead,AutoTokenizer
from neural_nlp.models.gpt_neox_model import GPTNeoXModel,GPTNeoXPosLearnedModel,GPTNeoXPosLearnedForCausalLM,GPTNeoXPosLearnedConfig,GPTNeoXConfig, initialize_gpt_neox_weights

def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples['text'],truncation=True, max_length=context_length, padding=True,return_overflowing_tokens=True,return_length=True)
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")

def evaluate(model):
    model.eval()
    losses = []
    for step, batch in tqdm(enumerate(eval_dataloader),total=len(eval_dataloader)):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])
        losses.append((outputs.loss))
    loss = torch.mean(torch.stack(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    print(f"validation loss: {loss.item()}, validation perplexity {perplexity.item()}")
    return loss.item(), perplexity.item()

context_length=1024
EVAL_BATCH_SIZE=32
if __name__ =='__main__':
    config_names=[x['weight_identifier'] for x in transformer_configurations]
    model_config=transformer_configurations[config_names.index('gpt2-neox-pos_learned-1B-v2-ckpnt-300000')]

    model_config = transformer_configurations[config_names.index('mistral/caprica-gpt2-small-x81/ckpt_190000')]
    model_conf = AutoConfig.from_pretrained(model_config['config_file'])
    model_ctr = AutoModelWithLMHead
    state_dict = None
    model = model_ctr.from_pretrained(model_config['weight_file'], config=model_conf, state_dict=state_dict)
    #list(model.state_dict().keys())
    model_conf=GPTNeoXPosLearnedConfig.from_pretrained(model_config['config_file'])
    model_ctr=GPTNeoXPosLearnedForCausalLM
    state_dict=None
    model = model_ctr.from_pretrained(model_config['weight_file'], config=model_conf, state_dict=state_dict)


    mini_dataset = load_dataset('/om/user/ehoseini/MyData/miniBERTa_v2', 'miniBERTa-10M')
    tokenizer = AutoTokenizer.from_pretrained("gpt2", fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_datasets = mini_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    print(f"Input IDs length: {len(tokenized_datasets['train']['input_ids'])}")
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=EVAL_BATCH_SIZE)

    (a,b)=evaluate(model)

