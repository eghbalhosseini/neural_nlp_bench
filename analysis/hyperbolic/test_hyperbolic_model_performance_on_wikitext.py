import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
from neural_nlp.models.implementations import transformer_configurations
import torch

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoConfig, AutoModel, AutoModelWithLMHead,AutoTokenizer, GPT2Config
import copy
import numpy as np

import torch


def symmetrize_GPT2(state_dict, sign='+', nheads=12, nlayers=12):
    #valid_keys=['attn.c_attn.weight', 'attn.c_attn.bias']
    valid_keys = ['transformer.h.1.attn.c_attn.weight',]
    if sign == '+':
        f = lambda x, y: x + y
    elif sign == '-':
        f = lambda x, y: x - y
    for keys in state_dict.keys():
        if any([x in keys for x in valid_keys]):
            if 'weight' in keys:
                weights=state_dict[keys]
                print(f'processing {keys}')
                Q, K, V = weights.split(weights.shape[0], dim=1)  # dimensions 768 x 768
                head_dim = weights.shape[0] // nheads
                Q = Q.split(head_dim, dim=0)  # dimensions 64 x 768
                K = K.split(head_dim, dim=0)  # dimensions 64 x 768
                newQ = []
                newK = []
                for head in range(nheads):
                    q, k = Q[head], K[head]  # dimensions 64 x 768
                    M = q @ k.T
                    symm = 0.5 * f(M, M.T)
                    fullq = torch.cat([symm] * nheads, axis=1) / nheads
                    fullk = torch.cat([torch.eye(M.shape[0])] * nheads, axis=1)
                    newQ.append(fullq)
                    newK.append(fullk)
                Q = torch.cat(newQ)
                K = torch.cat(newK)
                new_weights = torch.vstack([Q, K, V])
                state_dict[keys] = new_weights.T
            elif 'bias' in keys:
                weights=state_dict[keys.replace('bias','weight')]
                biases=state_dict[keys]
                qb, kb, v = biases.split(weights.shape[0], dim=0)
                new_biases = 0.5 * f(kb, qb)
                bias = torch.hstack([new_biases, new_biases, v])
                state_dict[keys] = bias
        else:
            state_dict[keys] = state_dict[keys]
    return state_dict

    # for layer in range(nlayers):
    #     if layer in []:
    #         weights = state_dict[f"transformer.h.{layer}.attn.c_attn.weight"]  # q,k,v #dimensions 768 x 2304
    #         biases = state_dict[f"transformer.h.{layer}.attn.c_attn.bias"]  # q,k,v
    #         Q, K, V = weights.split(weights.shape[0], dim=1)  # dimensions 768 x 768
    #         head_dim = weights.shape[0] // nheads
    #         Q = Q.split(head_dim, dim=0)  # dimensions 64 x 768
    #         K = K.split(head_dim, dim=0)  # dimensions 64 x 768
    #         newQ = []
    #         newK = []
    #         for head in range(nheads):
    #             q, k = Q[head], K[head]  # dimensions 64 x 768
    #             M = q @ k.T
    #             symm = 0.5 * f(M, M.T)
    #             fullq = torch.cat([symm] * nheads, axis=1) / nheads
    #             fullk = torch.cat([torch.eye(M.shape[0])] * nheads, axis=1)
    #             newQ.append(fullq)
    #             newK.append(fullk)
    #
    #         Q = torch.cat(newQ)
    #         K = torch.cat(newK)
    #         new_weights = torch.vstack([Q, K, V])
    #         state_dict[f"transformer.h.{layer}.attn.c_attn.weight"] = new_weights.T
    #         qb, kb, v = biases.split(weights.shape[0], dim=0)
    #         new_biases = 0.5 * f(kb, qb)
    #         bias = torch.hstack([new_biases, new_biases, v])
    #         state_dict[f"transformer.h.{layer}.attn.c_attn.bias"] = bias
    #     else:
    #         continue
    # return state_dict

def permute_mat(mat):
    mat_flat = mat.flatten()
    assert (mat_flat.ndim == 1)
    shuffle_idx = torch.randperm(mat_flat.shape[0])
    mat_flat_rnd = mat_flat[shuffle_idx]
    mat_perm = torch.reshape(mat_flat_rnd, mat.shape)
    return mat_perm

def initialize_gpt2_weights(model, mu=0.0, sigma=0.02, permute=False, valid_keys=None):
    model_perm = copy.deepcopy(model)
    orig_states = model_perm.state_dict()
    if valid_keys is None:
        valid_keys = ['attn.c_attn.weight', 'attn.c_attn.bias', 'attn.c_proj', 'ln_1', 'ln_2', 'mlp.c_fc', 'mlp.c_proj',
                      'wte', 'wpe', 'lm_head']
    if type(mu) is float:
        # make a dictionalry of mu and sigma for each key
        mu_dict = dict.fromkeys(valid_keys, mu)

    elif type(mu) is dict:
        # add the missing keys to the mu_dict
        remaining_keys = [x for x in valid_keys if x not in mu.keys()]
        mu_dict = {**mu, **dict.fromkeys(remaining_keys, 0.0)}
    else:
        raise ValueError('mu must be either float or dict')
    if type(sigma) is float:
        # make a dictionalry of mu and sigma for each key
        sigma_dict = dict.fromkeys(valid_keys, sigma)
    elif type(sigma) is dict:
        # add the missing keys to the mu_dict
        remaining_keys = [x for x in valid_keys if x not in sigma.keys()]
        sigma_dict = {**sigma, **dict.fromkeys(remaining_keys, 0.0)}
    else:
        raise ValueError('sigma must be either float or dict')

    to_permute = np.sum(
        [np.sum([valid_keys[n] in s for s in list(orig_states.keys())]) for n in range(len(valid_keys))])
    if permute:
        pbar = tqdm(total=to_permute, desc=f'permuting {to_permute} weights in {len(orig_states.keys())}')
    else:
        pbar = tqdm(total=to_permute, desc=f'initializing {to_permute} weights in {len(orig_states.keys())}')
    perm_states = dict.fromkeys(orig_states.keys())
    for key in orig_states.keys():
        if any([x in key for x in valid_keys]):
            a = orig_states[key]
            idx = [x in key for x in valid_keys].index(True)
            mu_key = valid_keys[idx]
            b = torch.normal(mu_dict[mu_key], sigma_dict[mu_key], size=a.shape)
            perm_states[key] = permute_mat(a) if permute else permute_mat(b)
            pbar.update()
        else:
            perm_states[key] = orig_states[key]

    return perm_states


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
    model_ctr=GPT2LMHeadModel
    model_config=GPT2Config.from_pretrained('gpt2')
    state_dict = None
    model = model_ctr.from_pretrained('gpt2', state_dict=state_dict)
    state_=model.state_dict()
    state_symmetric=symmetrize_GPT2(state_)

    model_sym = model_ctr.from_pretrained('gpt2',config=model_config, state_dict=state_symmetric)
    model_sym.to(device)

    #model_id = "gpt2"
    #model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    max_len = model.config.n_positions
    max_length=max_len
    stride = 512
    batch_size = 16


    #test = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    ppl_batch = batched_perplexity(model_sym, test, tokenizer, batch_size, stride)
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
