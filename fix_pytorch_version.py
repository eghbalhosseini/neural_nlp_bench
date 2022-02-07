# import numpy as np
import os
import argparse
import torch
parser = argparse.ArgumentParser(description='')
parser.add_argument('ckpt_dir', type=str, default='',
                    help='subject ID, e.g. "sub190" to run this script on')
args = parser.parse_args()
if __name__ == '__main__':
    ckpt_dir = args.ckpt_dir
    model_w = torch.load(f"{ckpt_dir}/pytorch_model.bin",map_location=torch.device('cpu'))
    torch.save(model_w, f"{ckpt_dir}/pytorch_model.bin",_use_new_zipfile_serialization=False)