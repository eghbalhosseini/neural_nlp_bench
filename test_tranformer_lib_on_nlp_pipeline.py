import argparse
from neural_nlp.models.implementations import _PytorchTransformerWrapper, word_last, transformer_configurations
from neural_nlp import score as score_function
import os
import numpy as np
import getpass

from transformers import GPT2Tokenizer
from transformers import GPT2Model
from transformers import GPT2Config

user=getpass.getuser()
if user=='eghbalhosseini':
    model_and_config_dir = '/Users/eghbalhosseini/MyData/arch_search'
elif user=='ehoseini':
    model_and_config_dir = '/om2/user/ehoseini/MyData/neural_nlp_bench/'

GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP = {"distilgpt2": os.path.join(model_and_config_dir,'CONFIG_ARCHIVE_MAP','originals','distilgpt2-config.json')}
GPT2_PRETRAINED_MODEL_ARCHIVE_MAP = {"distilgpt2": os.path.join(model_and_config_dir,'MODEL_ARCHIVE_MAP','distilgpt2-pytorch_model.bin')}

if __name__ =='__main__':
    #benchmark_name="Pereira2018-encoding"
    benchmark_name = 'Fedorenko2016v3-encoding'
    model_name="distilgpt2"
    config_file=GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP[model_name]
    model_file=GPT2_PRETRAINED_MODEL_ARCHIVE_MAP[model_name]
    benchmark_tsk = benchmark_name

    config = GPT2Config.from_json_file(config_file)
    num_layers = config.n_layer
    config.output_hidden_states = True
    config.state_dict = None
    model = GPT2Model(config)
    model.from_pretrained(model_file, config=config)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model_identifier = config.weight_identifier
    # find model index in model_configs
    config_idx = int(np.argwhere([x['weight_identifier'] == config.weight_identifier for x in transformer_configurations]))
    brainscore_config = transformer_configurations[config_idx]
    # - tokenizer_ctr: the importable class name of the model's tokenizer class
    brainscore_config['tokenizer_ctr'] = brainscore_config.get('tokenizer_ctr',brainscore_config['prefix'] + 'Tokenizer')
    # - model_ctr: the importable class name of the model's model class
    brainscore_config['model_ctr'] = brainscore_config.get('model_ctr', brainscore_config['prefix'] + 'Model')
    # - config_ctr: the importable class name of the model's config class
    brainscore_config['config_ctr'] = brainscore_config.get('config_ctr', brainscore_config['prefix'] + 'Config')

    transformer = _PytorchTransformerWrapper(identifier=model_identifier,
                                             tokenizer=tokenizer,
                                             tokenizer_special_tokens=brainscore_config.get('tokenizer_special_tokens',()),
                                             model=model,
                                             layers=list(brainscore_config['layers']),
                                             sentence_average=word_last)

    #score_results=score_function(benchmark=benchmark_tsk, model=model_identifier, model_impl=transformer,
    #                  layers=list(brainscore_config['layers']))
    score_results=score_function(benchmark=benchmark_tsk, model=model_identifier, model_impl=transformer,
                   layers=[list(brainscore_config['layers'])[-1]])