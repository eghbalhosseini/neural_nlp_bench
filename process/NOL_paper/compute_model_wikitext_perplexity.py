import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

from evaluate import load
import evaluate
import datasets
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler

from neural_nlp.models.implementations import transformer_configurations
import transformers
from transformers import AutoTokenizer
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoConfig, AutoModel, AutoModelWithLMHead,AutoTokenizer
from neural_nlp.models.gpt_neox_model import GPTNeoXModel,GPTNeoXPosLearnedModel,GPTNeoXPosLearnedForCausalLM,GPTNeoXPosLearnedConfig,GPTNeoXConfig, initialize_gpt_neox_weights
from neural_nlp.models import model_pool,initialize_gpt2_weights

def batched_perplexity(model, dataset, tokenizer, batch_size, stride):
    device = model.device
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    text_len = encodings.input_ids.size(1)
    lls = []

    for i in tqdm(range(0, text_len, batch_size * stride)):
        begin_locs, end_locs, trg_lens = [], [], []
        for j in range(batch_size):
            j = i + j * stride
            if j >= text_len:
                break
            begin_loc = max(j + stride - max_len, 0)
            end_loc = min(j + stride, text_len)
            trg_len = end_loc - j  # may be different from stride on last loop

            begin_locs.append(begin_loc)
            end_locs.append(end_loc)
            trg_lens.append(trg_len)

        input_ids = [encodings.input_ids[:, b:e] for b, e in zip(begin_locs, end_locs)]
        target_end_locs = [sen.size(-1) for sen in input_ids]
        input_ids = [
            F.pad(sen, (0, max_len - sen.size(-1)), "constant", 0) for sen in input_ids
        ] # we dont need attention mask as long as these padded token is not involved in loss calculation
        input_ids = torch.stack(input_ids, dim=1).squeeze(0).to(device)

        target_ids = torch.ones_like(input_ids) * -100 # -100 is the default ingore_index value in torch.nn.CrossEntropyLoss
        for i, (b, e) in enumerate(zip(trg_lens, target_end_locs)):
            labels = input_ids[i, -b:e].clone()
            target_ids[i, -b:e] = labels

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs["loss"] * sum(trg_lens)

        lls.append(log_likelihood)

    ppl = torch.exp(sum(torch.stack(lls) / end_locs[-1]))
    return ppl
context_length=1024
EVAL_BATCH_SIZE=32

if __name__ == "__main__":
    device = "cuda"
    #device='cpu'
    config_names = [x['weight_identifier'] for x in transformer_configurations]
    model_config = transformer_configurations[config_names.index('mistral-caprica-gpt2-small-x81-ckpnt-0')]


    model_conf = AutoConfig.from_pretrained(model_config['config_file'])
    model_ctr = AutoModelWithLMHead

    state_dict = None
    model = model_ctr.from_pretrained(model_config['weight_file'], config=model_conf, state_dict=state_dict)
    model.state_dict().keys()
    state_dict_permute=initialize_gpt2_weights(model,permute=False)
    model = model_ctr.from_pretrained(model_config['weight_file'], config=model_conf, state_dict=state_dict_permute)
    model.to(device)

    #model_id = "gpt2"
    #model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_conf.model_type)
    max_len = model.config.n_positions
    max_length=max_len
    stride = 512
    batch_size = 16


    #test = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    ppl_batch = batched_perplexity(model, test, tokenizer, batch_size, stride)
    print(f"--------------{ppl_batch=}-------------")


    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    nlls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc).cpu()

    print(f"--------------{ppl=}-------------")







# model_config=transformer_configurations[config_names.index('gpt2-neox-pos_learned-1B-v2-ckpnt-300000')]
    #    model_conf = GPTNeoXPosLearnedConfig.from_pretrained(model_config['config_file'])
    #    model_ctr = GPTNeoXPosLearnedForCausalLM
    #    model = model_ctr(config=model_conf)
    #    state_dict = None
    #    model = model_ctr.from_pretrained(model_config['weight_file'], config=model_conf, state_dict=state_dict)

#    perplexity = evaluate.load("perplexity", module_type="metric")
#    results = perplexity.compute(model_id='gpt2', add_start_token=False, input_texts=input_texts)
# print(round(results["mean_perplexity"], 2))

#perplexity = evaluate.load("perplexity", module_type="metric")

    # input_text = datasets.load_dataset("wikitext",
    #                                     "wikitext-2-raw-v1",
    #                                     split="test")["text"]
    # input_texts = [s for s in input_text if s != '']
