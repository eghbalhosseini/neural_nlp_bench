# import numpy as np
import os
import argparse
import torch
from pathlib import Path
import glob
from tqdm import tqdm
parser = argparse.ArgumentParser(description='')
parser.add_argument('--ckpt_dir', type=str, default='')
parser.add_argument('--ckpt_type', type=str, default='mistral') # either mistral or gpt_neox
args = parser.parse_args()
if __name__ == '__main__':
    ckpt_dir = args.ckpt_dir
    ckpt_type= args.ckpt_type
    if ckpt_type=='mistral':
        model_w = torch.load(f"{ckpt_dir}/pytorch_model.bin",map_location=torch.device('cpu'))
        torch.save(model_w, f"{ckpt_dir}/pytorch_model.bin",_use_new_zipfile_serialization=False)
    elif ckpt_type=='gpt_neox':
        # make a directory for 1.5 versions and below
        ckpt_pth=Path(ckpt_dir)
        ckpt_pth.parent
        new_path=Path(ckpt_pth.parent,ckpt_pth.parts[-1]+'_torch_1_5')
        if new_path.exists():
            pass
        else:
            os.mkdir(str(new_path))
        # go through the files in original path and transform them into new ones
        pt_files=glob.glob(str(ckpt_pth)+'/*.pt')
        for pt_file in tqdm(pt_files):
            pt_path=Path(pt_file)
            pt_path.name
            layer_file=torch.load(pt_file,map_location=torch.device('cpu'))
            torch.save(layer_file,os.path.join(str(new_path),pt_path.name),_use_new_zipfile_serialization=False)
