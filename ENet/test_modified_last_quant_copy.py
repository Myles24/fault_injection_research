# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os,sys
import shutil
if os.environ["W_QUANT"]=='1':
    # load quant apis
    from pytorch_nndct.apis import torch_quantizer, dump_xmodel

import random
import torch
import numpy as np
from torch.autograd import Variable
from torch.backends import cudnn
import torch.nn.functional as F


cudnn.benchmark = True
import torch.nn as nn
import logging
import argparse
from torchsummary import summary
from time import gmtime, strftime 
from fault_correction import get_corrected_data, get_free_data, get_faulty_data
import matplotlib.pyplot as plt

def copy_part_of_state_dict(source_state_dict, target_state_dict, prefix, replace=False):
    """
    Copies a portion of weights from a source state_dict to a target state_dict.

    Args:
        source_state_dict (dict): The source state_dict to copy from.
        target_state_dict (dict): The target state_dict to copy to.
        prefix (str): The prefix of the keys to copy.
        replace (bool, optional): If True, replace existing keys in target_state_dict. 
                                  If False, only copy if the key doesn't exist.
    """
    for key in source_state_dict:
        if key.startswith(prefix):
            target_key = key
            if replace or target_key not in target_state_dict:
                target_state_dict[target_key] = source_state_dict[key].clone()
                #print(f"Copied {key} to {target_key}")
            else:
                #print(f"Skipped {key} as {target_key} already exists.")
                pass


class Configs():
    def __init__(self):
        parser = argparse.ArgumentParser("ENet(modified) on Cityscapes")
        #dataset options
        parser.add_argument('--dataset', type=str, default='cityscapes', help='dataset name')
        parser.add_argument('--data_root', type=str, default='./data/cityscapes', help='path to dataset')
        parser.add_argument('--num_classes', type=int, default=19, help='classes numbers')
        parser.add_argument('--ignore_label', type=int, default=255, help='ignore index')

        parser.add_argument('--checkpoint_dir', type=str, default='ckpt-cityscapes', help='path to checkpoint')
        parser.add_argument('--input_size', nargs='+', type=int, default=[1024, 512], help='input size')
        parser.add_argument('--weight', type=str, default=None, help='resume from weight')
        parser.add_argument('--test_only', action='store_true', help='if only test the trained model')
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--gpu_num', type=int, default=1, help='number of gpus')
        parser.add_argument('--print_freq', type=int, default=20, help='print frequency')
        #validation options
        parser.add_argument('--val_batch_size', type=int, default=1, help='batch size')
        # evaluation miou options
        parser.add_argument('--eval', action='store_true', help='evaluation miou mode')
        # demo options
        parser.add_argument('--demo_dir', type=str, default='./data/demo/', help='path to demo dataset')
        parser.add_argument('--save_dir', type=str, default='./data/demo_results', help='path to save demo prediction')

        parser.add_argument('--quant_dir', type=str, default='quantize_result', help='path to save quant info')
        parser.add_argument('--quant_mode', default='calib', choices=['float', 'calib', 'test'], \
                                            help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
        parser.add_argument('--finetune', dest='finetune', action='store_true', help='finetune model before calibration')
        parser.add_argument('--dump_xmodel', dest='dump_xmodel', action='store_true', help='dump xmodel after test')
        parser.add_argument('--device', default='cpu', choices=['gpu', 'cpu'], help='assign runtime device')

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args

class Criterion(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, weight=None, use_weight=True, reduce=True):
        super(Criterion, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        loss = self.criterion(pred, target)
        return loss


def main(args):

    if args.dump_xmodel:
        args.device='cpu'
        args.val_batch_size=1

    if args.device=='cpu':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    # network
    from code.models.enet_xilinx_modified import ENet, ENet_last
    #from code.models.enet_xilinx_modified_last import ENet_last
    print('====> Bulid Networks...')
    net = ENet(args.num_classes).to(device)
    #net_modified = ENet_modified(args.num_classes).to(device)
    net_last = ENet_last(args.num_classes).to(device)


    # pytorch_total_params = sum(p.numel() for p in net_modified.parameters())
    # print("Total number of parameters",pytorch_total_params)
    
    #pytorch_total_params = sum(p.numel() for p in net.parameters() if net.requires_grad)
    #print("Total number of trainable parameters",pytorch_total_params)
    if os.path.isfile(args.weight):
        state_dict = torch.load(args.weight, map_location=device) 

        keys_to_add = [
        "upsample4_0_main.main_conv.weight",
        "upsample4_0_main.main_bn.weight",
        "upsample4_0_main.main_bn.bias",
        "upsample4_0_main.main_bn.running_mean",
        "upsample4_0_main.main_bn.running_var"
        ]
        # Build the new keys by modifying the existing key names
        for new_key in keys_to_add:
            # Remove the "_main" from the new key to get the original key
            original_key = new_key.replace("_main.", ".", 1)

            if original_key in state_dict['state_dict']:
                state_dict['state_dict'][new_key] = state_dict['state_dict'][original_key].clone()
                print(f"Copied {original_key} → {new_key}")
            else:
                print(f"Original key {original_key} not found in state_dict")
        to_delete = ["regular4_1.ext1_conv.weight", "regular4_1.ext1_bn.weight", "regular4_1.ext1_bn.bias", "regular4_1.ext1_bn.running_mean", "regular4_1.ext1_bn.running_var", "regular4_1.ext1_bn.num_batches_tracked", "regular4_1.ext2_conv.weight", "regular4_1.ext2_bn.weight", "regular4_1.ext2_bn.bias", "regular4_1.ext2_bn.running_mean", "regular4_1.ext2_bn.running_var", "regular4_1.ext2_bn.num_batches_tracked", "regular4_1.ext3_conv.weight", "regular4_1.ext3_bn.weight", "regular4_1.ext3_bn.bias", "regular4_1.ext3_bn.running_mean", "regular4_1.ext3_bn.running_var", "regular4_1.ext3_bn.num_batches_tracked", "regular4_2.ext1_conv.weight", "regular4_2.ext1_bn.weight", "regular4_2.ext1_bn.bias", "regular4_2.ext1_bn.running_mean", "regular4_2.ext1_bn.running_var", "regular4_2.ext1_bn.num_batches_tracked", "regular4_2.ext2_conv.weight", "regular4_2.ext2_bn.weight", "regular4_2.ext2_bn.bias", "regular4_2.ext2_bn.running_mean", "regular4_2.ext2_bn.running_var", "regular4_2.ext2_bn.num_batches_tracked", "regular4_2.ext3_conv.weight", "regular4_2.ext3_bn.weight", "regular4_2.ext3_bn.bias", "regular4_2.ext3_bn.running_mean", "regular4_2.ext3_bn.running_var", "regular4_2.ext3_bn.num_batches_tracked", "upsample5_0.main_conv.weight", "upsample5_0.main_bn.weight", "upsample5_0.main_bn.bias", "upsample5_0.main_bn.running_mean", "upsample5_0.main_bn.running_var", "upsample5_0.main_bn.num_batches_tracked", "upsample5_0.ext1_conv.weight", "upsample5_0.ext1_bn.weight", "upsample5_0.ext1_bn.bias", "upsample5_0.ext1_bn.running_mean", "upsample5_0.ext1_bn.running_var", "upsample5_0.ext1_bn.num_batches_tracked", "upsample5_0.ext2_conv.weight", "upsample5_0.ext2_bn.weight", "upsample5_0.ext2_bn.bias", "upsample5_0.ext2_bn.running_mean", "upsample5_0.ext2_bn.running_var", "upsample5_0.ext2_bn.num_batches_tracked", "upsample5_0.ext3_conv.weight", "upsample5_0.ext3_bn.weight", "upsample5_0.ext3_bn.bias", "upsample5_0.ext3_bn.running_mean", "upsample5_0.ext3_bn.running_var", "upsample5_0.ext3_bn.num_batches_tracked", "regular5_1.ext1_conv.weight", "regular5_1.ext1_bn.weight", "regular5_1.ext1_bn.bias", "regular5_1.ext1_bn.running_mean", "regular5_1.ext1_bn.running_var", "regular5_1.ext1_bn.num_batches_tracked", "regular5_1.ext2_conv.weight", "regular5_1.ext2_bn.weight", "regular5_1.ext2_bn.bias", "regular5_1.ext2_bn.running_mean", "regular5_1.ext2_bn.running_var", "regular5_1.ext2_bn.num_batches_tracked", "regular5_1.ext3_conv.weight", "regular5_1.ext3_bn.weight", "regular5_1.ext3_bn.bias", "regular5_1.ext3_bn.running_mean", "regular5_1.ext3_bn.running_var", "regular5_1.ext3_bn.num_batches_tracked", "transposed_conv.weight", "upsample4_0.main_conv.weight", "upsample4_0.main_bn.weight", "upsample4_0.main_bn.bias", "upsample4_0.main_bn.running_mean", "upsample4_0.main_bn.running_var", "upsample4_0.main_bn.num_batches_tracked", "upsample4_0.ext2_conv.weight", "upsample4_0.ext2_bn.weight", "upsample4_0.ext2_bn.bias", "upsample4_0.ext2_bn.running_mean", "upsample4_0.ext2_bn.running_var", "upsample4_0.ext2_bn.num_batches_tracked", "upsample4_0.ext3_conv.weight", "upsample4_0.ext3_bn.weight", "upsample4_0.ext3_bn.bias", "upsample4_0.ext3_bn.running_mean", "upsample4_0.ext3_bn.running_var", "upsample4_0.ext3_bn.num_batches_tracked"]
        #stage 4 weights
        #to_delete= ["upsample5_0.main_conv.weight", "upsample5_0.main_bn.weight", "upsample5_0.main_bn.bias", "upsample5_0.main_bn.running_mean", "upsample5_0.main_bn.running_var", "upsample5_0.main_bn.num_batches_tracked", "upsample5_0.ext1_conv.weight", "upsample5_0.ext1_bn.weight", "upsample5_0.ext1_bn.bias", "upsample5_0.ext1_bn.running_mean", "upsample5_0.ext1_bn.running_var", "upsample5_0.ext1_bn.num_batches_tracked", "upsample5_0.ext2_conv.weight", "upsample5_0.ext2_bn.weight", "upsample5_0.ext2_bn.bias", "upsample5_0.ext2_bn.running_mean", "upsample5_0.ext2_bn.running_var", "upsample5_0.ext2_bn.num_batches_tracked", "upsample5_0.ext3_conv.weight", "upsample5_0.ext3_bn.weight", "upsample5_0.ext3_bn.bias", "upsample5_0.ext3_bn.running_mean", "upsample5_0.ext3_bn.running_var", "upsample5_0.ext3_bn.num_batches_tracked", "regular5_1.ext1_conv.weight", "regular5_1.ext1_bn.weight", "regular5_1.ext1_bn.bias", "regular5_1.ext1_bn.running_mean", "regular5_1.ext1_bn.running_var", "regular5_1.ext1_bn.num_batches_tracked", "regular5_1.ext2_conv.weight", "regular5_1.ext2_bn.weight", "regular5_1.ext2_bn.bias", "regular5_1.ext2_bn.running_mean", "regular5_1.ext2_bn.running_var", "regular5_1.ext2_bn.num_batches_tracked", "regular5_1.ext3_conv.weight", "regular5_1.ext3_bn.weight", "regular5_1.ext3_bn.bias", "regular5_1.ext3_bn.running_mean", "regular5_1.ext3_bn.running_var", "regular5_1.ext3_bn.num_batches_tracked", "transposed_conv.weight"]
        # stage 4 intralayer
        #to_delete= ["upsample5_0.main_conv.weight", "upsample5_0.main_bn.weight", "upsample5_0.main_bn.bias", "upsample5_0.main_bn.running_mean", "upsample5_0.main_bn.running_var", "upsample5_0.main_bn.num_batches_tracked", "upsample5_0.ext1_conv.weight", "upsample5_0.ext1_bn.weight", "upsample5_0.ext1_bn.bias", "upsample5_0.ext1_bn.running_mean", "upsample5_0.ext1_bn.running_var", "upsample5_0.ext1_bn.num_batches_tracked", "upsample5_0.ext2_conv.weight", "upsample5_0.ext2_bn.weight", "upsample5_0.ext2_bn.bias", "upsample5_0.ext2_bn.running_mean", "upsample5_0.ext2_bn.running_var", "upsample5_0.ext2_bn.num_batches_tracked", "upsample5_0.ext3_conv.weight", "upsample5_0.ext3_bn.weight", "upsample5_0.ext3_bn.bias", "upsample5_0.ext3_bn.running_mean", "upsample5_0.ext3_bn.running_var", "upsample5_0.ext3_bn.num_batches_tracked", "regular5_1.ext1_conv.weight", "regular5_1.ext1_bn.weight", "regular5_1.ext1_bn.bias", "regular5_1.ext1_bn.running_mean", "regular5_1.ext1_bn.running_var", "regular5_1.ext1_bn.num_batches_tracked", "regular5_1.ext2_conv.weight", "regular5_1.ext2_bn.weight", "regular5_1.ext2_bn.bias", "regular5_1.ext2_bn.running_mean", "regular5_1.ext2_bn.running_var", "regular5_1.ext2_bn.num_batches_tracked", "regular5_1.ext3_conv.weight", "regular5_1.ext3_bn.weight", "regular5_1.ext3_bn.bias", "regular5_1.ext3_bn.running_mean", "regular5_1.ext3_bn.running_var", "regular5_1.ext3_bn.num_batches_tracked", "transposed_conv.weight", "regular4_2.ext1_conv.weight", "regular4_2.ext1_bn.weight", "regular4_2.ext1_bn.bias", "regular4_2.ext1_bn.running_mean", "regular4_2.ext1_bn.running_var", "regular4_2.ext1_bn.num_batches_tracked", "regular4_2.ext2_conv.weight", "regular4_2.ext2_bn.weight", "regular4_2.ext2_bn.bias", "regular4_2.ext2_bn.running_mean", "regular4_2.ext2_bn.running_var", "regular4_2.ext2_bn.num_batches_tracked", "regular4_2.ext3_conv.weight", "regular4_2.ext3_bn.weight", "regular4_2.ext3_bn.bias", "regular4_2.ext3_bn.running_mean", "regular4_2.ext3_bn.running_var", "regular4_2.ext3_bn.num_batches_tracked"]
        #stage 1 weights
        #to_delete = ["downsample2_0.main_conv.weight", "downsample2_0.main_bn.weight", "downsample2_0.main_bn.bias", "downsample2_0.main_bn.running_mean", "downsample2_0.main_bn.running_var", "downsample2_0.main_bn.num_batches_tracked", "downsample2_0.ext1_conv.weight", "downsample2_0.ext1_bn.weight", "downsample2_0.ext1_bn.bias", "downsample2_0.ext1_bn.running_mean", "downsample2_0.ext1_bn.running_var", "downsample2_0.ext1_bn.num_batches_tracked", "downsample2_0.ext2_conv.weight", "downsample2_0.ext2_bn.weight", "downsample2_0.ext2_bn.bias", "downsample2_0.ext2_bn.running_mean", "downsample2_0.ext2_bn.running_var", "downsample2_0.ext2_bn.num_batches_tracked", "downsample2_0.ext3_conv.weight", "downsample2_0.ext3_bn.weight", "downsample2_0.ext3_bn.bias", "downsample2_0.ext3_bn.running_mean", "downsample2_0.ext3_bn.running_var", "downsample2_0.ext3_bn.num_batches_tracked", "regular2_1.ext1_conv.weight", "regular2_1.ext1_bn.weight", "regular2_1.ext1_bn.bias", "regular2_1.ext1_bn.running_mean", "regular2_1.ext1_bn.running_var", "regular2_1.ext1_bn.num_batches_tracked", "regular2_1.ext2_conv.weight", "regular2_1.ext2_bn.weight", "regular2_1.ext2_bn.bias", "regular2_1.ext2_bn.running_mean", "regular2_1.ext2_bn.running_var", "regular2_1.ext2_bn.num_batches_tracked", "regular2_1.ext3_conv.weight", "regular2_1.ext3_bn.weight", "regular2_1.ext3_bn.bias", "regular2_1.ext3_bn.running_mean", "regular2_1.ext3_bn.running_var", "regular2_1.ext3_bn.num_batches_tracked", "dilated2_2.ext1_conv.weight", "dilated2_2.ext1_bn.weight", "dilated2_2.ext1_bn.bias", "dilated2_2.ext1_bn.running_mean", "dilated2_2.ext1_bn.running_var", "dilated2_2.ext1_bn.num_batches_tracked", "dilated2_2.ext2_conv.weight", "dilated2_2.ext2_bn.weight", "dilated2_2.ext2_bn.bias", "dilated2_2.ext2_bn.running_mean", "dilated2_2.ext2_bn.running_var", "dilated2_2.ext2_bn.num_batches_tracked", "dilated2_2.ext3_conv.weight", "dilated2_2.ext3_bn.weight", "dilated2_2.ext3_bn.bias", "dilated2_2.ext3_bn.running_mean", "dilated2_2.ext3_bn.running_var", "dilated2_2.ext3_bn.num_batches_tracked", "asymmetric2_3.ext1_conv.weight", "asymmetric2_3.ext1_bn.weight", "asymmetric2_3.ext1_bn.bias", "asymmetric2_3.ext1_bn.running_mean", "asymmetric2_3.ext1_bn.running_var", "asymmetric2_3.ext1_bn.num_batches_tracked", "asymmetric2_3.ext2_conv.weight", "asymmetric2_3.ext2_bn.weight", "asymmetric2_3.ext2_bn.bias", "asymmetric2_3.ext2_bn.running_mean", "asymmetric2_3.ext2_bn.running_var", "asymmetric2_3.ext2_bn.num_batches_tracked", "asymmetric2_3.ext3_conv.weight", "asymmetric2_3.ext3_bn.weight", "asymmetric2_3.ext3_bn.bias", "asymmetric2_3.ext3_bn.running_mean", "asymmetric2_3.ext3_bn.running_var", "asymmetric2_3.ext3_bn.num_batches_tracked", "dilated2_4.ext1_conv.weight", "dilated2_4.ext1_bn.weight", "dilated2_4.ext1_bn.bias", "dilated2_4.ext1_bn.running_mean", "dilated2_4.ext1_bn.running_var", "dilated2_4.ext1_bn.num_batches_tracked", "dilated2_4.ext2_conv.weight", "dilated2_4.ext2_bn.weight", "dilated2_4.ext2_bn.bias", "dilated2_4.ext2_bn.running_mean", "dilated2_4.ext2_bn.running_var", "dilated2_4.ext2_bn.num_batches_tracked", "dilated2_4.ext3_conv.weight", "dilated2_4.ext3_bn.weight", "dilated2_4.ext3_bn.bias", "dilated2_4.ext3_bn.running_mean", "dilated2_4.ext3_bn.running_var", "dilated2_4.ext3_bn.num_batches_tracked", "regular2_5.ext1_conv.weight", "regular2_5.ext1_bn.weight", "regular2_5.ext1_bn.bias", "regular2_5.ext1_bn.running_mean", "regular2_5.ext1_bn.running_var", "regular2_5.ext1_bn.num_batches_tracked", "regular2_5.ext2_conv.weight", "regular2_5.ext2_bn.weight", "regular2_5.ext2_bn.bias", "regular2_5.ext2_bn.running_mean", "regular2_5.ext2_bn.running_var", "regular2_5.ext2_bn.num_batches_tracked", "regular2_5.ext3_conv.weight", "regular2_5.ext3_bn.weight", "regular2_5.ext3_bn.bias", "regular2_5.ext3_bn.running_mean", "regular2_5.ext3_bn.running_var", "regular2_5.ext3_bn.num_batches_tracked", "dilated2_6.ext1_conv.weight", "dilated2_6.ext1_bn.weight", "dilated2_6.ext1_bn.bias", "dilated2_6.ext1_bn.running_mean", "dilated2_6.ext1_bn.running_var", "dilated2_6.ext1_bn.num_batches_tracked", "dilated2_6.ext2_conv.weight", "dilated2_6.ext2_bn.weight", "dilated2_6.ext2_bn.bias", "dilated2_6.ext2_bn.running_mean", "dilated2_6.ext2_bn.running_var", "dilated2_6.ext2_bn.num_batches_tracked", "dilated2_6.ext3_conv.weight", "dilated2_6.ext3_bn.weight", "dilated2_6.ext3_bn.bias", "dilated2_6.ext3_bn.running_mean", "dilated2_6.ext3_bn.running_var", "dilated2_6.ext3_bn.num_batches_tracked", "asymmetric2_7.ext1_conv.weight", "asymmetric2_7.ext1_bn.weight", "asymmetric2_7.ext1_bn.bias", "asymmetric2_7.ext1_bn.running_mean", "asymmetric2_7.ext1_bn.running_var", "asymmetric2_7.ext1_bn.num_batches_tracked", "asymmetric2_7.ext2_conv.weight", "asymmetric2_7.ext2_bn.weight", "asymmetric2_7.ext2_bn.bias", "asymmetric2_7.ext2_bn.running_mean", "asymmetric2_7.ext2_bn.running_var", "asymmetric2_7.ext2_bn.num_batches_tracked", "asymmetric2_7.ext3_conv.weight", "asymmetric2_7.ext3_bn.weight", "asymmetric2_7.ext3_bn.bias", "asymmetric2_7.ext3_bn.running_mean", "asymmetric2_7.ext3_bn.running_var", "asymmetric2_7.ext3_bn.num_batches_tracked", "dilated2_8.ext1_conv.weight", "dilated2_8.ext1_bn.weight", "dilated2_8.ext1_bn.bias", "dilated2_8.ext1_bn.running_mean", "dilated2_8.ext1_bn.running_var", "dilated2_8.ext1_bn.num_batches_tracked", "dilated2_8.ext2_conv.weight", "dilated2_8.ext2_bn.weight", "dilated2_8.ext2_bn.bias", "dilated2_8.ext2_bn.running_mean", "dilated2_8.ext2_bn.running_var", "dilated2_8.ext2_bn.num_batches_tracked", "dilated2_8.ext3_conv.weight", "dilated2_8.ext3_bn.weight", "dilated2_8.ext3_bn.bias", "dilated2_8.ext3_bn.running_mean", "dilated2_8.ext3_bn.running_var", "dilated2_8.ext3_bn.num_batches_tracked", "regular3_0.ext1_conv.weight", "regular3_0.ext1_bn.weight", "regular3_0.ext1_bn.bias", "regular3_0.ext1_bn.running_mean", "regular3_0.ext1_bn.running_var", "regular3_0.ext1_bn.num_batches_tracked", "regular3_0.ext2_conv.weight", "regular3_0.ext2_bn.weight", "regular3_0.ext2_bn.bias", "regular3_0.ext2_bn.running_mean", "regular3_0.ext2_bn.running_var", "regular3_0.ext2_bn.num_batches_tracked", "regular3_0.ext3_conv.weight", "regular3_0.ext3_bn.weight", "regular3_0.ext3_bn.bias", "regular3_0.ext3_bn.running_mean", "regular3_0.ext3_bn.running_var", "regular3_0.ext3_bn.num_batches_tracked", "dilated3_1.ext1_conv.weight", "dilated3_1.ext1_bn.weight", "dilated3_1.ext1_bn.bias", "dilated3_1.ext1_bn.running_mean", "dilated3_1.ext1_bn.running_var", "dilated3_1.ext1_bn.num_batches_tracked", "dilated3_1.ext2_conv.weight", "dilated3_1.ext2_bn.weight", "dilated3_1.ext2_bn.bias", "dilated3_1.ext2_bn.running_mean", "dilated3_1.ext2_bn.running_var", "dilated3_1.ext2_bn.num_batches_tracked", "dilated3_1.ext3_conv.weight", "dilated3_1.ext3_bn.weight", "dilated3_1.ext3_bn.bias", "dilated3_1.ext3_bn.running_mean", "dilated3_1.ext3_bn.running_var", "dilated3_1.ext3_bn.num_batches_tracked", "asymmetric3_2.ext1_conv.weight", "asymmetric3_2.ext1_bn.weight", "asymmetric3_2.ext1_bn.bias", "asymmetric3_2.ext1_bn.running_mean", "asymmetric3_2.ext1_bn.running_var", "asymmetric3_2.ext1_bn.num_batches_tracked", "asymmetric3_2.ext2_conv.weight", "asymmetric3_2.ext2_bn.weight", "asymmetric3_2.ext2_bn.bias", "asymmetric3_2.ext2_bn.running_mean", "asymmetric3_2.ext2_bn.running_var", "asymmetric3_2.ext2_bn.num_batches_tracked", "asymmetric3_2.ext3_conv.weight", "asymmetric3_2.ext3_bn.weight", "asymmetric3_2.ext3_bn.bias", "asymmetric3_2.ext3_bn.running_mean", "asymmetric3_2.ext3_bn.running_var", "asymmetric3_2.ext3_bn.num_batches_tracked", "dilated3_3.ext1_conv.weight", "dilated3_3.ext1_bn.weight", "dilated3_3.ext1_bn.bias", "dilated3_3.ext1_bn.running_mean", "dilated3_3.ext1_bn.running_var", "dilated3_3.ext1_bn.num_batches_tracked", "dilated3_3.ext2_conv.weight", "dilated3_3.ext2_bn.weight", "dilated3_3.ext2_bn.bias", "dilated3_3.ext2_bn.running_mean", "dilated3_3.ext2_bn.running_var", "dilated3_3.ext2_bn.num_batches_tracked", "dilated3_3.ext3_conv.weight", "dilated3_3.ext3_bn.weight", "dilated3_3.ext3_bn.bias", "dilated3_3.ext3_bn.running_mean", "dilated3_3.ext3_bn.running_var", "dilated3_3.ext3_bn.num_batches_tracked", "regular3_4.ext1_conv.weight", "regular3_4.ext1_bn.weight", "regular3_4.ext1_bn.bias", "regular3_4.ext1_bn.running_mean", "regular3_4.ext1_bn.running_var", "regular3_4.ext1_bn.num_batches_tracked", "regular3_4.ext2_conv.weight", "regular3_4.ext2_bn.weight", "regular3_4.ext2_bn.bias", "regular3_4.ext2_bn.running_mean", "regular3_4.ext2_bn.running_var", "regular3_4.ext2_bn.num_batches_tracked", "regular3_4.ext3_conv.weight", "regular3_4.ext3_bn.weight", "regular3_4.ext3_bn.bias", "regular3_4.ext3_bn.running_mean", "regular3_4.ext3_bn.running_var", "regular3_4.ext3_bn.num_batches_tracked", "dilated3_5.ext1_conv.weight", "dilated3_5.ext1_bn.weight", "dilated3_5.ext1_bn.bias", "dilated3_5.ext1_bn.running_mean", "dilated3_5.ext1_bn.running_var", "dilated3_5.ext1_bn.num_batches_tracked", "dilated3_5.ext2_conv.weight", "dilated3_5.ext2_bn.weight", "dilated3_5.ext2_bn.bias", "dilated3_5.ext2_bn.running_mean", "dilated3_5.ext2_bn.running_var", "dilated3_5.ext2_bn.num_batches_tracked", "dilated3_5.ext3_conv.weight", "dilated3_5.ext3_bn.weight", "dilated3_5.ext3_bn.bias", "dilated3_5.ext3_bn.running_mean", "dilated3_5.ext3_bn.running_var", "dilated3_5.ext3_bn.num_batches_tracked", "asymmetric3_6.ext1_conv.weight", "asymmetric3_6.ext1_bn.weight", "asymmetric3_6.ext1_bn.bias", "asymmetric3_6.ext1_bn.running_mean", "asymmetric3_6.ext1_bn.running_var", "asymmetric3_6.ext1_bn.num_batches_tracked", "asymmetric3_6.ext2_conv.weight", "asymmetric3_6.ext2_bn.weight", "asymmetric3_6.ext2_bn.bias", "asymmetric3_6.ext2_bn.running_mean", "asymmetric3_6.ext2_bn.running_var", "asymmetric3_6.ext2_bn.num_batches_tracked", "asymmetric3_6.ext3_conv.weight", "asymmetric3_6.ext3_bn.weight", "asymmetric3_6.ext3_bn.bias", "asymmetric3_6.ext3_bn.running_mean", "asymmetric3_6.ext3_bn.running_var", "asymmetric3_6.ext3_bn.num_batches_tracked", "dilated3_7.ext1_conv.weight", "dilated3_7.ext1_bn.weight", "dilated3_7.ext1_bn.bias", "dilated3_7.ext1_bn.running_mean", "dilated3_7.ext1_bn.running_var", "dilated3_7.ext1_bn.num_batches_tracked", "dilated3_7.ext2_conv.weight", "dilated3_7.ext2_bn.weight", "dilated3_7.ext2_bn.bias", "dilated3_7.ext2_bn.running_mean", "dilated3_7.ext2_bn.running_var", "dilated3_7.ext2_bn.num_batches_tracked", "dilated3_7.ext3_conv.weight", "dilated3_7.ext3_bn.weight", "dilated3_7.ext3_bn.bias", "dilated3_7.ext3_bn.running_mean", "dilated3_7.ext3_bn.running_var", "dilated3_7.ext3_bn.num_batches_tracked", "upsample4_0.main_conv.weight", "upsample4_0.main_bn.weight", "upsample4_0.main_bn.bias", "upsample4_0.main_bn.running_mean", "upsample4_0.main_bn.running_var", "upsample4_0.main_bn.num_batches_tracked", "upsample4_0.ext1_conv.weight", "upsample4_0.ext1_bn.weight", "upsample4_0.ext1_bn.bias", "upsample4_0.ext1_bn.running_mean", "upsample4_0.ext1_bn.running_var", "upsample4_0.ext1_bn.num_batches_tracked", "upsample4_0.ext2_conv.weight", "upsample4_0.ext2_bn.weight", "upsample4_0.ext2_bn.bias", "upsample4_0.ext2_bn.running_mean", "upsample4_0.ext2_bn.running_var", "upsample4_0.ext2_bn.num_batches_tracked", "upsample4_0.ext3_conv.weight", "upsample4_0.ext3_bn.weight", "upsample4_0.ext3_bn.bias", "upsample4_0.ext3_bn.running_mean", "upsample4_0.ext3_bn.running_var", "upsample4_0.ext3_bn.num_batches_tracked", "regular4_1.ext1_conv.weight", "regular4_1.ext1_bn.weight", "regular4_1.ext1_bn.bias", "regular4_1.ext1_bn.running_mean", "regular4_1.ext1_bn.running_var", "regular4_1.ext1_bn.num_batches_tracked", "regular4_1.ext2_conv.weight", "regular4_1.ext2_bn.weight", "regular4_1.ext2_bn.bias", "regular4_1.ext2_bn.running_mean", "regular4_1.ext2_bn.running_var", "regular4_1.ext2_bn.num_batches_tracked", "regular4_1.ext3_conv.weight", "regular4_1.ext3_bn.weight", "regular4_1.ext3_bn.bias", "regular4_1.ext3_bn.running_mean", "regular4_1.ext3_bn.running_var", "regular4_1.ext3_bn.num_batches_tracked", "regular4_2.ext1_conv.weight", "regular4_2.ext1_bn.weight", "regular4_2.ext1_bn.bias", "regular4_2.ext1_bn.running_mean", "regular4_2.ext1_bn.running_var", "regular4_2.ext1_bn.num_batches_tracked", "regular4_2.ext2_conv.weight", "regular4_2.ext2_bn.weight", "regular4_2.ext2_bn.bias", "regular4_2.ext2_bn.running_mean", "regular4_2.ext2_bn.running_var", "regular4_2.ext2_bn.num_batches_tracked", "regular4_2.ext3_conv.weight", "regular4_2.ext3_bn.weight", "regular4_2.ext3_bn.bias", "regular4_2.ext3_bn.running_mean", "regular4_2.ext3_bn.running_var", "regular4_2.ext3_bn.num_batches_tracked", "upsample5_0.main_conv.weight", "upsample5_0.main_bn.weight", "upsample5_0.main_bn.bias", "upsample5_0.main_bn.running_mean", "upsample5_0.main_bn.running_var", "upsample5_0.main_bn.num_batches_tracked", "upsample5_0.ext1_conv.weight", "upsample5_0.ext1_bn.weight", "upsample5_0.ext1_bn.bias", "upsample5_0.ext1_bn.running_mean", "upsample5_0.ext1_bn.running_var", "upsample5_0.ext1_bn.num_batches_tracked", "upsample5_0.ext2_conv.weight", "upsample5_0.ext2_bn.weight", "upsample5_0.ext2_bn.bias", "upsample5_0.ext2_bn.running_mean", "upsample5_0.ext2_bn.running_var", "upsample5_0.ext2_bn.num_batches_tracked", "upsample5_0.ext3_conv.weight", "upsample5_0.ext3_bn.weight", "upsample5_0.ext3_bn.bias", "upsample5_0.ext3_bn.running_mean", "upsample5_0.ext3_bn.running_var", "upsample5_0.ext3_bn.num_batches_tracked", "regular5_1.ext1_conv.weight", "regular5_1.ext1_bn.weight", "regular5_1.ext1_bn.bias", "regular5_1.ext1_bn.running_mean", "regular5_1.ext1_bn.running_var", "regular5_1.ext1_bn.num_batches_tracked", "regular5_1.ext2_conv.weight", "regular5_1.ext2_bn.weight", "regular5_1.ext2_bn.bias", "regular5_1.ext2_bn.running_mean", "regular5_1.ext2_bn.running_var", "regular5_1.ext2_bn.num_batches_tracked", "regular5_1.ext3_conv.weight", "regular5_1.ext3_bn.weight", "regular5_1.ext3_bn.bias", "regular5_1.ext3_bn.running_mean", "regular5_1.ext3_bn.running_var", "regular5_1.ext3_bn.num_batches_tracked", "transposed_conv.weight"]
        #stage 1_regular_1_3 weights
        #to_delete = ["downsample2_0.main_conv.weight", "downsample2_0.main_bn.weight", "downsample2_0.main_bn.bias", "downsample2_0.main_bn.running_mean", "downsample2_0.main_bn.running_var", "downsample2_0.main_bn.num_batches_tracked", "downsample2_0.ext1_conv.weight", "downsample2_0.ext1_bn.weight", "downsample2_0.ext1_bn.bias", "downsample2_0.ext1_bn.running_mean", "downsample2_0.ext1_bn.running_var", "downsample2_0.ext1_bn.num_batches_tracked", "downsample2_0.ext2_conv.weight", "downsample2_0.ext2_bn.weight", "downsample2_0.ext2_bn.bias", "downsample2_0.ext2_bn.running_mean", "downsample2_0.ext2_bn.running_var", "downsample2_0.ext2_bn.num_batches_tracked", "downsample2_0.ext3_conv.weight", "downsample2_0.ext3_bn.weight", "downsample2_0.ext3_bn.bias", "downsample2_0.ext3_bn.running_mean", "downsample2_0.ext3_bn.running_var", "downsample2_0.ext3_bn.num_batches_tracked", "regular2_1.ext1_conv.weight", "regular2_1.ext1_bn.weight", "regular2_1.ext1_bn.bias", "regular2_1.ext1_bn.running_mean", "regular2_1.ext1_bn.running_var", "regular2_1.ext1_bn.num_batches_tracked", "regular2_1.ext2_conv.weight", "regular2_1.ext2_bn.weight", "regular2_1.ext2_bn.bias", "regular2_1.ext2_bn.running_mean", "regular2_1.ext2_bn.running_var", "regular2_1.ext2_bn.num_batches_tracked", "regular2_1.ext3_conv.weight", "regular2_1.ext3_bn.weight", "regular2_1.ext3_bn.bias", "regular2_1.ext3_bn.running_mean", "regular2_1.ext3_bn.running_var", "regular2_1.ext3_bn.num_batches_tracked", "dilated2_2.ext1_conv.weight", "dilated2_2.ext1_bn.weight", "dilated2_2.ext1_bn.bias", "dilated2_2.ext1_bn.running_mean", "dilated2_2.ext1_bn.running_var", "dilated2_2.ext1_bn.num_batches_tracked", "dilated2_2.ext2_conv.weight", "dilated2_2.ext2_bn.weight", "dilated2_2.ext2_bn.bias", "dilated2_2.ext2_bn.running_mean", "dilated2_2.ext2_bn.running_var", "dilated2_2.ext2_bn.num_batches_tracked", "dilated2_2.ext3_conv.weight", "dilated2_2.ext3_bn.weight", "dilated2_2.ext3_bn.bias", "dilated2_2.ext3_bn.running_mean", "dilated2_2.ext3_bn.running_var", "dilated2_2.ext3_bn.num_batches_tracked", "asymmetric2_3.ext1_conv.weight", "asymmetric2_3.ext1_bn.weight", "asymmetric2_3.ext1_bn.bias", "asymmetric2_3.ext1_bn.running_mean", "asymmetric2_3.ext1_bn.running_var", "asymmetric2_3.ext1_bn.num_batches_tracked", "asymmetric2_3.ext2_conv.weight", "asymmetric2_3.ext2_bn.weight", "asymmetric2_3.ext2_bn.bias", "asymmetric2_3.ext2_bn.running_mean", "asymmetric2_3.ext2_bn.running_var", "asymmetric2_3.ext2_bn.num_batches_tracked", "asymmetric2_3.ext3_conv.weight", "asymmetric2_3.ext3_bn.weight", "asymmetric2_3.ext3_bn.bias", "asymmetric2_3.ext3_bn.running_mean", "asymmetric2_3.ext3_bn.running_var", "asymmetric2_3.ext3_bn.num_batches_tracked", "dilated2_4.ext1_conv.weight", "dilated2_4.ext1_bn.weight", "dilated2_4.ext1_bn.bias", "dilated2_4.ext1_bn.running_mean", "dilated2_4.ext1_bn.running_var", "dilated2_4.ext1_bn.num_batches_tracked", "dilated2_4.ext2_conv.weight", "dilated2_4.ext2_bn.weight", "dilated2_4.ext2_bn.bias", "dilated2_4.ext2_bn.running_mean", "dilated2_4.ext2_bn.running_var", "dilated2_4.ext2_bn.num_batches_tracked", "dilated2_4.ext3_conv.weight", "dilated2_4.ext3_bn.weight", "dilated2_4.ext3_bn.bias", "dilated2_4.ext3_bn.running_mean", "dilated2_4.ext3_bn.running_var", "dilated2_4.ext3_bn.num_batches_tracked", "regular2_5.ext1_conv.weight", "regular2_5.ext1_bn.weight", "regular2_5.ext1_bn.bias", "regular2_5.ext1_bn.running_mean", "regular2_5.ext1_bn.running_var", "regular2_5.ext1_bn.num_batches_tracked", "regular2_5.ext2_conv.weight", "regular2_5.ext2_bn.weight", "regular2_5.ext2_bn.bias", "regular2_5.ext2_bn.running_mean", "regular2_5.ext2_bn.running_var", "regular2_5.ext2_bn.num_batches_tracked", "regular2_5.ext3_conv.weight", "regular2_5.ext3_bn.weight", "regular2_5.ext3_bn.bias", "regular2_5.ext3_bn.running_mean", "regular2_5.ext3_bn.running_var", "regular2_5.ext3_bn.num_batches_tracked", "dilated2_6.ext1_conv.weight", "dilated2_6.ext1_bn.weight", "dilated2_6.ext1_bn.bias", "dilated2_6.ext1_bn.running_mean", "dilated2_6.ext1_bn.running_var", "dilated2_6.ext1_bn.num_batches_tracked", "dilated2_6.ext2_conv.weight", "dilated2_6.ext2_bn.weight", "dilated2_6.ext2_bn.bias", "dilated2_6.ext2_bn.running_mean", "dilated2_6.ext2_bn.running_var", "dilated2_6.ext2_bn.num_batches_tracked", "dilated2_6.ext3_conv.weight", "dilated2_6.ext3_bn.weight", "dilated2_6.ext3_bn.bias", "dilated2_6.ext3_bn.running_mean", "dilated2_6.ext3_bn.running_var", "dilated2_6.ext3_bn.num_batches_tracked", "asymmetric2_7.ext1_conv.weight", "asymmetric2_7.ext1_bn.weight", "asymmetric2_7.ext1_bn.bias", "asymmetric2_7.ext1_bn.running_mean", "asymmetric2_7.ext1_bn.running_var", "asymmetric2_7.ext1_bn.num_batches_tracked", "asymmetric2_7.ext2_conv.weight", "asymmetric2_7.ext2_bn.weight", "asymmetric2_7.ext2_bn.bias", "asymmetric2_7.ext2_bn.running_mean", "asymmetric2_7.ext2_bn.running_var", "asymmetric2_7.ext2_bn.num_batches_tracked", "asymmetric2_7.ext3_conv.weight", "asymmetric2_7.ext3_bn.weight", "asymmetric2_7.ext3_bn.bias", "asymmetric2_7.ext3_bn.running_mean", "asymmetric2_7.ext3_bn.running_var", "asymmetric2_7.ext3_bn.num_batches_tracked", "dilated2_8.ext1_conv.weight", "dilated2_8.ext1_bn.weight", "dilated2_8.ext1_bn.bias", "dilated2_8.ext1_bn.running_mean", "dilated2_8.ext1_bn.running_var", "dilated2_8.ext1_bn.num_batches_tracked", "dilated2_8.ext2_conv.weight", "dilated2_8.ext2_bn.weight", "dilated2_8.ext2_bn.bias", "dilated2_8.ext2_bn.running_mean", "dilated2_8.ext2_bn.running_var", "dilated2_8.ext2_bn.num_batches_tracked", "dilated2_8.ext3_conv.weight", "dilated2_8.ext3_bn.weight", "dilated2_8.ext3_bn.bias", "dilated2_8.ext3_bn.running_mean", "dilated2_8.ext3_bn.running_var", "dilated2_8.ext3_bn.num_batches_tracked", "regular3_0.ext1_conv.weight", "regular3_0.ext1_bn.weight", "regular3_0.ext1_bn.bias", "regular3_0.ext1_bn.running_mean", "regular3_0.ext1_bn.running_var", "regular3_0.ext1_bn.num_batches_tracked", "regular3_0.ext2_conv.weight", "regular3_0.ext2_bn.weight", "regular3_0.ext2_bn.bias", "regular3_0.ext2_bn.running_mean", "regular3_0.ext2_bn.running_var", "regular3_0.ext2_bn.num_batches_tracked", "regular3_0.ext3_conv.weight", "regular3_0.ext3_bn.weight", "regular3_0.ext3_bn.bias", "regular3_0.ext3_bn.running_mean", "regular3_0.ext3_bn.running_var", "regular3_0.ext3_bn.num_batches_tracked", "dilated3_1.ext1_conv.weight", "dilated3_1.ext1_bn.weight", "dilated3_1.ext1_bn.bias", "dilated3_1.ext1_bn.running_mean", "dilated3_1.ext1_bn.running_var", "dilated3_1.ext1_bn.num_batches_tracked", "dilated3_1.ext2_conv.weight", "dilated3_1.ext2_bn.weight", "dilated3_1.ext2_bn.bias", "dilated3_1.ext2_bn.running_mean", "dilated3_1.ext2_bn.running_var", "dilated3_1.ext2_bn.num_batches_tracked", "dilated3_1.ext3_conv.weight", "dilated3_1.ext3_bn.weight", "dilated3_1.ext3_bn.bias", "dilated3_1.ext3_bn.running_mean", "dilated3_1.ext3_bn.running_var", "dilated3_1.ext3_bn.num_batches_tracked", "asymmetric3_2.ext1_conv.weight", "asymmetric3_2.ext1_bn.weight", "asymmetric3_2.ext1_bn.bias", "asymmetric3_2.ext1_bn.running_mean", "asymmetric3_2.ext1_bn.running_var", "asymmetric3_2.ext1_bn.num_batches_tracked", "asymmetric3_2.ext2_conv.weight", "asymmetric3_2.ext2_bn.weight", "asymmetric3_2.ext2_bn.bias", "asymmetric3_2.ext2_bn.running_mean", "asymmetric3_2.ext2_bn.running_var", "asymmetric3_2.ext2_bn.num_batches_tracked", "asymmetric3_2.ext3_conv.weight", "asymmetric3_2.ext3_bn.weight", "asymmetric3_2.ext3_bn.bias", "asymmetric3_2.ext3_bn.running_mean", "asymmetric3_2.ext3_bn.running_var", "asymmetric3_2.ext3_bn.num_batches_tracked", "dilated3_3.ext1_conv.weight", "dilated3_3.ext1_bn.weight", "dilated3_3.ext1_bn.bias", "dilated3_3.ext1_bn.running_mean", "dilated3_3.ext1_bn.running_var", "dilated3_3.ext1_bn.num_batches_tracked", "dilated3_3.ext2_conv.weight", "dilated3_3.ext2_bn.weight", "dilated3_3.ext2_bn.bias", "dilated3_3.ext2_bn.running_mean", "dilated3_3.ext2_bn.running_var", "dilated3_3.ext2_bn.num_batches_tracked", "dilated3_3.ext3_conv.weight", "dilated3_3.ext3_bn.weight", "dilated3_3.ext3_bn.bias", "dilated3_3.ext3_bn.running_mean", "dilated3_3.ext3_bn.running_var", "dilated3_3.ext3_bn.num_batches_tracked", "regular3_4.ext1_conv.weight", "regular3_4.ext1_bn.weight", "regular3_4.ext1_bn.bias", "regular3_4.ext1_bn.running_mean", "regular3_4.ext1_bn.running_var", "regular3_4.ext1_bn.num_batches_tracked", "regular3_4.ext2_conv.weight", "regular3_4.ext2_bn.weight", "regular3_4.ext2_bn.bias", "regular3_4.ext2_bn.running_mean", "regular3_4.ext2_bn.running_var", "regular3_4.ext2_bn.num_batches_tracked", "regular3_4.ext3_conv.weight", "regular3_4.ext3_bn.weight", "regular3_4.ext3_bn.bias", "regular3_4.ext3_bn.running_mean", "regular3_4.ext3_bn.running_var", "regular3_4.ext3_bn.num_batches_tracked", "dilated3_5.ext1_conv.weight", "dilated3_5.ext1_bn.weight", "dilated3_5.ext1_bn.bias", "dilated3_5.ext1_bn.running_mean", "dilated3_5.ext1_bn.running_var", "dilated3_5.ext1_bn.num_batches_tracked", "dilated3_5.ext2_conv.weight", "dilated3_5.ext2_bn.weight", "dilated3_5.ext2_bn.bias", "dilated3_5.ext2_bn.running_mean", "dilated3_5.ext2_bn.running_var", "dilated3_5.ext2_bn.num_batches_tracked", "dilated3_5.ext3_conv.weight", "dilated3_5.ext3_bn.weight", "dilated3_5.ext3_bn.bias", "dilated3_5.ext3_bn.running_mean", "dilated3_5.ext3_bn.running_var", "dilated3_5.ext3_bn.num_batches_tracked", "asymmetric3_6.ext1_conv.weight", "asymmetric3_6.ext1_bn.weight", "asymmetric3_6.ext1_bn.bias", "asymmetric3_6.ext1_bn.running_mean", "asymmetric3_6.ext1_bn.running_var", "asymmetric3_6.ext1_bn.num_batches_tracked", "asymmetric3_6.ext2_conv.weight", "asymmetric3_6.ext2_bn.weight", "asymmetric3_6.ext2_bn.bias", "asymmetric3_6.ext2_bn.running_mean", "asymmetric3_6.ext2_bn.running_var", "asymmetric3_6.ext2_bn.num_batches_tracked", "asymmetric3_6.ext3_conv.weight", "asymmetric3_6.ext3_bn.weight", "asymmetric3_6.ext3_bn.bias", "asymmetric3_6.ext3_bn.running_mean", "asymmetric3_6.ext3_bn.running_var", "asymmetric3_6.ext3_bn.num_batches_tracked", "dilated3_7.ext1_conv.weight", "dilated3_7.ext1_bn.weight", "dilated3_7.ext1_bn.bias", "dilated3_7.ext1_bn.running_mean", "dilated3_7.ext1_bn.running_var", "dilated3_7.ext1_bn.num_batches_tracked", "dilated3_7.ext2_conv.weight", "dilated3_7.ext2_bn.weight", "dilated3_7.ext2_bn.bias", "dilated3_7.ext2_bn.running_mean", "dilated3_7.ext2_bn.running_var", "dilated3_7.ext2_bn.num_batches_tracked", "dilated3_7.ext3_conv.weight", "dilated3_7.ext3_bn.weight", "dilated3_7.ext3_bn.bias", "dilated3_7.ext3_bn.running_mean", "dilated3_7.ext3_bn.running_var", "dilated3_7.ext3_bn.num_batches_tracked", "upsample4_0.main_conv.weight", "upsample4_0.main_bn.weight", "upsample4_0.main_bn.bias", "upsample4_0.main_bn.running_mean", "upsample4_0.main_bn.running_var", "upsample4_0.main_bn.num_batches_tracked", "upsample4_0.ext1_conv.weight", "upsample4_0.ext1_bn.weight", "upsample4_0.ext1_bn.bias", "upsample4_0.ext1_bn.running_mean", "upsample4_0.ext1_bn.running_var", "upsample4_0.ext1_bn.num_batches_tracked", "upsample4_0.ext2_conv.weight", "upsample4_0.ext2_bn.weight", "upsample4_0.ext2_bn.bias", "upsample4_0.ext2_bn.running_mean", "upsample4_0.ext2_bn.running_var", "upsample4_0.ext2_bn.num_batches_tracked", "upsample4_0.ext3_conv.weight", "upsample4_0.ext3_bn.weight", "upsample4_0.ext3_bn.bias", "upsample4_0.ext3_bn.running_mean", "upsample4_0.ext3_bn.running_var", "upsample4_0.ext3_bn.num_batches_tracked", "regular4_1.ext1_conv.weight", "regular4_1.ext1_bn.weight", "regular4_1.ext1_bn.bias", "regular4_1.ext1_bn.running_mean", "regular4_1.ext1_bn.running_var", "regular4_1.ext1_bn.num_batches_tracked", "regular4_1.ext2_conv.weight", "regular4_1.ext2_bn.weight", "regular4_1.ext2_bn.bias", "regular4_1.ext2_bn.running_mean", "regular4_1.ext2_bn.running_var", "regular4_1.ext2_bn.num_batches_tracked", "regular4_1.ext3_conv.weight", "regular4_1.ext3_bn.weight", "regular4_1.ext3_bn.bias", "regular4_1.ext3_bn.running_mean", "regular4_1.ext3_bn.running_var", "regular4_1.ext3_bn.num_batches_tracked", "regular4_2.ext1_conv.weight", "regular4_2.ext1_bn.weight", "regular4_2.ext1_bn.bias", "regular4_2.ext1_bn.running_mean", "regular4_2.ext1_bn.running_var", "regular4_2.ext1_bn.num_batches_tracked", "regular4_2.ext2_conv.weight", "regular4_2.ext2_bn.weight", "regular4_2.ext2_bn.bias", "regular4_2.ext2_bn.running_mean", "regular4_2.ext2_bn.running_var", "regular4_2.ext2_bn.num_batches_tracked", "regular4_2.ext3_conv.weight", "regular4_2.ext3_bn.weight", "regular4_2.ext3_bn.bias", "regular4_2.ext3_bn.running_mean", "regular4_2.ext3_bn.running_var", "regular4_2.ext3_bn.num_batches_tracked", "upsample5_0.main_conv.weight", "upsample5_0.main_bn.weight", "upsample5_0.main_bn.bias", "upsample5_0.main_bn.running_mean", "upsample5_0.main_bn.running_var", "upsample5_0.main_bn.num_batches_tracked", "upsample5_0.ext1_conv.weight", "upsample5_0.ext1_bn.weight", "upsample5_0.ext1_bn.bias", "upsample5_0.ext1_bn.running_mean", "upsample5_0.ext1_bn.running_var", "upsample5_0.ext1_bn.num_batches_tracked", "upsample5_0.ext2_conv.weight", "upsample5_0.ext2_bn.weight", "upsample5_0.ext2_bn.bias", "upsample5_0.ext2_bn.running_mean", "upsample5_0.ext2_bn.running_var", "upsample5_0.ext2_bn.num_batches_tracked", "upsample5_0.ext3_conv.weight", "upsample5_0.ext3_bn.weight", "upsample5_0.ext3_bn.bias", "upsample5_0.ext3_bn.running_mean", "upsample5_0.ext3_bn.running_var", "upsample5_0.ext3_bn.num_batches_tracked", "regular5_1.ext1_conv.weight", "regular5_1.ext1_bn.weight", "regular5_1.ext1_bn.bias", "regular5_1.ext1_bn.running_mean", "regular5_1.ext1_bn.running_var", "regular5_1.ext1_bn.num_batches_tracked", "regular5_1.ext2_conv.weight", "regular5_1.ext2_bn.weight", "regular5_1.ext2_bn.bias", "regular5_1.ext2_bn.running_mean", "regular5_1.ext2_bn.running_var", "regular5_1.ext2_bn.num_batches_tracked", "regular5_1.ext3_conv.weight", "regular5_1.ext3_bn.weight", "regular5_1.ext3_bn.bias", "regular5_1.ext3_bn.running_mean", "regular5_1.ext3_bn.running_var", "regular5_1.ext3_bn.num_batches_tracked", "transposed_conv.weight", "regular1_4.ext1_conv.weight", "regular1_4.ext1_bn.weight", "regular1_4.ext1_bn.bias", "regular1_4.ext1_bn.running_mean", "regular1_4.ext1_bn.running_var", "regular1_4.ext1_bn.num_batches_tracked", "regular1_4.ext2_conv.weight", "regular1_4.ext2_bn.weight", "regular1_4.ext2_bn.bias", "regular1_4.ext2_bn.running_mean", "regular1_4.ext2_bn.running_var", "regular1_4.ext2_bn.num_batches_tracked", "regular1_4.ext3_conv.weight", "regular1_4.ext3_bn.weight", "regular1_4.ext3_bn.bias", "regular1_4.ext3_bn.running_mean", "regular1_4.ext3_bn.running_var", "regular1_4.ext3_bn.num_batches_tracked"]
        #stage 2 weights
        #to_delete = ["regular3_0.ext1_conv.weight", "regular3_0.ext1_bn.weight", "regular3_0.ext1_bn.bias", "regular3_0.ext1_bn.running_mean", "regular3_0.ext1_bn.running_var", "regular3_0.ext1_bn.num_batches_tracked", "regular3_0.ext2_conv.weight", "regular3_0.ext2_bn.weight", "regular3_0.ext2_bn.bias", "regular3_0.ext2_bn.running_mean", "regular3_0.ext2_bn.running_var", "regular3_0.ext2_bn.num_batches_tracked", "regular3_0.ext3_conv.weight", "regular3_0.ext3_bn.weight", "regular3_0.ext3_bn.bias", "regular3_0.ext3_bn.running_mean", "regular3_0.ext3_bn.running_var", "regular3_0.ext3_bn.num_batches_tracked", "dilated3_1.ext1_conv.weight", "dilated3_1.ext1_bn.weight", "dilated3_1.ext1_bn.bias", "dilated3_1.ext1_bn.running_mean", "dilated3_1.ext1_bn.running_var", "dilated3_1.ext1_bn.num_batches_tracked", "dilated3_1.ext2_conv.weight", "dilated3_1.ext2_bn.weight", "dilated3_1.ext2_bn.bias", "dilated3_1.ext2_bn.running_mean", "dilated3_1.ext2_bn.running_var", "dilated3_1.ext2_bn.num_batches_tracked", "dilated3_1.ext3_conv.weight", "dilated3_1.ext3_bn.weight", "dilated3_1.ext3_bn.bias", "dilated3_1.ext3_bn.running_mean", "dilated3_1.ext3_bn.running_var", "dilated3_1.ext3_bn.num_batches_tracked", "asymmetric3_2.ext1_conv.weight", "asymmetric3_2.ext1_bn.weight", "asymmetric3_2.ext1_bn.bias", "asymmetric3_2.ext1_bn.running_mean", "asymmetric3_2.ext1_bn.running_var", "asymmetric3_2.ext1_bn.num_batches_tracked", "asymmetric3_2.ext2_conv.weight", "asymmetric3_2.ext2_bn.weight", "asymmetric3_2.ext2_bn.bias", "asymmetric3_2.ext2_bn.running_mean", "asymmetric3_2.ext2_bn.running_var", "asymmetric3_2.ext2_bn.num_batches_tracked", "asymmetric3_2.ext3_conv.weight", "asymmetric3_2.ext3_bn.weight", "asymmetric3_2.ext3_bn.bias", "asymmetric3_2.ext3_bn.running_mean", "asymmetric3_2.ext3_bn.running_var", "asymmetric3_2.ext3_bn.num_batches_tracked", "dilated3_3.ext1_conv.weight", "dilated3_3.ext1_bn.weight", "dilated3_3.ext1_bn.bias", "dilated3_3.ext1_bn.running_mean", "dilated3_3.ext1_bn.running_var", "dilated3_3.ext1_bn.num_batches_tracked", "dilated3_3.ext2_conv.weight", "dilated3_3.ext2_bn.weight", "dilated3_3.ext2_bn.bias", "dilated3_3.ext2_bn.running_mean", "dilated3_3.ext2_bn.running_var", "dilated3_3.ext2_bn.num_batches_tracked", "dilated3_3.ext3_conv.weight", "dilated3_3.ext3_bn.weight", "dilated3_3.ext3_bn.bias", "dilated3_3.ext3_bn.running_mean", "dilated3_3.ext3_bn.running_var", "dilated3_3.ext3_bn.num_batches_tracked", "regular3_4.ext1_conv.weight", "regular3_4.ext1_bn.weight", "regular3_4.ext1_bn.bias", "regular3_4.ext1_bn.running_mean", "regular3_4.ext1_bn.running_var", "regular3_4.ext1_bn.num_batches_tracked", "regular3_4.ext2_conv.weight", "regular3_4.ext2_bn.weight", "regular3_4.ext2_bn.bias", "regular3_4.ext2_bn.running_mean", "regular3_4.ext2_bn.running_var", "regular3_4.ext2_bn.num_batches_tracked", "regular3_4.ext3_conv.weight", "regular3_4.ext3_bn.weight", "regular3_4.ext3_bn.bias", "regular3_4.ext3_bn.running_mean", "regular3_4.ext3_bn.running_var", "regular3_4.ext3_bn.num_batches_tracked", "dilated3_5.ext1_conv.weight", "dilated3_5.ext1_bn.weight", "dilated3_5.ext1_bn.bias", "dilated3_5.ext1_bn.running_mean", "dilated3_5.ext1_bn.running_var", "dilated3_5.ext1_bn.num_batches_tracked", "dilated3_5.ext2_conv.weight", "dilated3_5.ext2_bn.weight", "dilated3_5.ext2_bn.bias", "dilated3_5.ext2_bn.running_mean", "dilated3_5.ext2_bn.running_var", "dilated3_5.ext2_bn.num_batches_tracked", "dilated3_5.ext3_conv.weight", "dilated3_5.ext3_bn.weight", "dilated3_5.ext3_bn.bias", "dilated3_5.ext3_bn.running_mean", "dilated3_5.ext3_bn.running_var", "dilated3_5.ext3_bn.num_batches_tracked", "asymmetric3_6.ext1_conv.weight", "asymmetric3_6.ext1_bn.weight", "asymmetric3_6.ext1_bn.bias", "asymmetric3_6.ext1_bn.running_mean", "asymmetric3_6.ext1_bn.running_var", "asymmetric3_6.ext1_bn.num_batches_tracked", "asymmetric3_6.ext2_conv.weight", "asymmetric3_6.ext2_bn.weight", "asymmetric3_6.ext2_bn.bias", "asymmetric3_6.ext2_bn.running_mean", "asymmetric3_6.ext2_bn.running_var", "asymmetric3_6.ext2_bn.num_batches_tracked", "asymmetric3_6.ext3_conv.weight", "asymmetric3_6.ext3_bn.weight", "asymmetric3_6.ext3_bn.bias", "asymmetric3_6.ext3_bn.running_mean", "asymmetric3_6.ext3_bn.running_var", "asymmetric3_6.ext3_bn.num_batches_tracked", "dilated3_7.ext1_conv.weight", "dilated3_7.ext1_bn.weight", "dilated3_7.ext1_bn.bias", "dilated3_7.ext1_bn.running_mean", "dilated3_7.ext1_bn.running_var", "dilated3_7.ext1_bn.num_batches_tracked", "dilated3_7.ext2_conv.weight", "dilated3_7.ext2_bn.weight", "dilated3_7.ext2_bn.bias", "dilated3_7.ext2_bn.running_mean", "dilated3_7.ext2_bn.running_var", "dilated3_7.ext2_bn.num_batches_tracked", "dilated3_7.ext3_conv.weight", "dilated3_7.ext3_bn.weight", "dilated3_7.ext3_bn.bias", "dilated3_7.ext3_bn.running_mean", "dilated3_7.ext3_bn.running_var", "dilated3_7.ext3_bn.num_batches_tracked", "upsample4_0.main_conv.weight", "upsample4_0.main_bn.weight", "upsample4_0.main_bn.bias", "upsample4_0.main_bn.running_mean", "upsample4_0.main_bn.running_var", "upsample4_0.main_bn.num_batches_tracked", "upsample4_0.ext1_conv.weight", "upsample4_0.ext1_bn.weight", "upsample4_0.ext1_bn.bias", "upsample4_0.ext1_bn.running_mean", "upsample4_0.ext1_bn.running_var", "upsample4_0.ext1_bn.num_batches_tracked", "upsample4_0.ext2_conv.weight", "upsample4_0.ext2_bn.weight", "upsample4_0.ext2_bn.bias", "upsample4_0.ext2_bn.running_mean", "upsample4_0.ext2_bn.running_var", "upsample4_0.ext2_bn.num_batches_tracked", "upsample4_0.ext3_conv.weight", "upsample4_0.ext3_bn.weight", "upsample4_0.ext3_bn.bias", "upsample4_0.ext3_bn.running_mean", "upsample4_0.ext3_bn.running_var", "upsample4_0.ext3_bn.num_batches_tracked", "regular4_1.ext1_conv.weight", "regular4_1.ext1_bn.weight", "regular4_1.ext1_bn.bias", "regular4_1.ext1_bn.running_mean", "regular4_1.ext1_bn.running_var", "regular4_1.ext1_bn.num_batches_tracked", "regular4_1.ext2_conv.weight", "regular4_1.ext2_bn.weight", "regular4_1.ext2_bn.bias", "regular4_1.ext2_bn.running_mean", "regular4_1.ext2_bn.running_var", "regular4_1.ext2_bn.num_batches_tracked", "regular4_1.ext3_conv.weight", "regular4_1.ext3_bn.weight", "regular4_1.ext3_bn.bias", "regular4_1.ext3_bn.running_mean", "regular4_1.ext3_bn.running_var", "regular4_1.ext3_bn.num_batches_tracked", "regular4_2.ext1_conv.weight", "regular4_2.ext1_bn.weight", "regular4_2.ext1_bn.bias", "regular4_2.ext1_bn.running_mean", "regular4_2.ext1_bn.running_var", "regular4_2.ext1_bn.num_batches_tracked", "regular4_2.ext2_conv.weight", "regular4_2.ext2_bn.weight", "regular4_2.ext2_bn.bias", "regular4_2.ext2_bn.running_mean", "regular4_2.ext2_bn.running_var", "regular4_2.ext2_bn.num_batches_tracked", "regular4_2.ext3_conv.weight", "regular4_2.ext3_bn.weight", "regular4_2.ext3_bn.bias", "regular4_2.ext3_bn.running_mean", "regular4_2.ext3_bn.running_var", "regular4_2.ext3_bn.num_batches_tracked", "upsample5_0.main_conv.weight", "upsample5_0.main_bn.weight", "upsample5_0.main_bn.bias", "upsample5_0.main_bn.running_mean", "upsample5_0.main_bn.running_var", "upsample5_0.main_bn.num_batches_tracked", "upsample5_0.ext1_conv.weight", "upsample5_0.ext1_bn.weight", "upsample5_0.ext1_bn.bias", "upsample5_0.ext1_bn.running_mean", "upsample5_0.ext1_bn.running_var", "upsample5_0.ext1_bn.num_batches_tracked", "upsample5_0.ext2_conv.weight", "upsample5_0.ext2_bn.weight", "upsample5_0.ext2_bn.bias", "upsample5_0.ext2_bn.running_mean", "upsample5_0.ext2_bn.running_var", "upsample5_0.ext2_bn.num_batches_tracked", "upsample5_0.ext3_conv.weight", "upsample5_0.ext3_bn.weight", "upsample5_0.ext3_bn.bias", "upsample5_0.ext3_bn.running_mean", "upsample5_0.ext3_bn.running_var", "upsample5_0.ext3_bn.num_batches_tracked", "regular5_1.ext1_conv.weight", "regular5_1.ext1_bn.weight", "regular5_1.ext1_bn.bias", "regular5_1.ext1_bn.running_mean", "regular5_1.ext1_bn.running_var", "regular5_1.ext1_bn.num_batches_tracked", "regular5_1.ext2_conv.weight", "regular5_1.ext2_bn.weight", "regular5_1.ext2_bn.bias", "regular5_1.ext2_bn.running_mean", "regular5_1.ext2_bn.running_var", "regular5_1.ext2_bn.num_batches_tracked", "regular5_1.ext3_conv.weight", "regular5_1.ext3_bn.weight", "regular5_1.ext3_bn.bias", "regular5_1.ext3_bn.running_mean", "regular5_1.ext3_bn.running_var", "regular5_1.ext3_bn.num_batches_tracked", "transposed_conv.weight"]
        #stage 3 weights
        #to_delete = ["upsample4_0.main_conv.weight", "upsample4_0.main_bn.weight", "upsample4_0.main_bn.bias", "upsample4_0.main_bn.running_mean", "upsample4_0.main_bn.running_var", "upsample4_0.main_bn.num_batches_tracked", "upsample4_0.ext1_conv.weight", "upsample4_0.ext1_bn.weight", "upsample4_0.ext1_bn.bias", "upsample4_0.ext1_bn.running_mean", "upsample4_0.ext1_bn.running_var", "upsample4_0.ext1_bn.num_batches_tracked", "upsample4_0.ext2_conv.weight", "upsample4_0.ext2_bn.weight", "upsample4_0.ext2_bn.bias", "upsample4_0.ext2_bn.running_mean", "upsample4_0.ext2_bn.running_var", "upsample4_0.ext2_bn.num_batches_tracked", "upsample4_0.ext3_conv.weight", "upsample4_0.ext3_bn.weight", "upsample4_0.ext3_bn.bias", "upsample4_0.ext3_bn.running_mean", "upsample4_0.ext3_bn.running_var", "upsample4_0.ext3_bn.num_batches_tracked", "regular4_1.ext1_conv.weight", "regular4_1.ext1_bn.weight", "regular4_1.ext1_bn.bias", "regular4_1.ext1_bn.running_mean", "regular4_1.ext1_bn.running_var", "regular4_1.ext1_bn.num_batches_tracked", "regular4_1.ext2_conv.weight", "regular4_1.ext2_bn.weight", "regular4_1.ext2_bn.bias", "regular4_1.ext2_bn.running_mean", "regular4_1.ext2_bn.running_var", "regular4_1.ext2_bn.num_batches_tracked", "regular4_1.ext3_conv.weight", "regular4_1.ext3_bn.weight", "regular4_1.ext3_bn.bias", "regular4_1.ext3_bn.running_mean", "regular4_1.ext3_bn.running_var", "regular4_1.ext3_bn.num_batches_tracked", "regular4_2.ext1_conv.weight", "regular4_2.ext1_bn.weight", "regular4_2.ext1_bn.bias", "regular4_2.ext1_bn.running_mean", "regular4_2.ext1_bn.running_var", "regular4_2.ext1_bn.num_batches_tracked", "regular4_2.ext2_conv.weight", "regular4_2.ext2_bn.weight", "regular4_2.ext2_bn.bias", "regular4_2.ext2_bn.running_mean", "regular4_2.ext2_bn.running_var", "regular4_2.ext2_bn.num_batches_tracked", "regular4_2.ext3_conv.weight", "regular4_2.ext3_bn.weight", "regular4_2.ext3_bn.bias", "regular4_2.ext3_bn.running_mean", "regular4_2.ext3_bn.running_var", "regular4_2.ext3_bn.num_batches_tracked", "upsample5_0.main_conv.weight", "upsample5_0.main_bn.weight", "upsample5_0.main_bn.bias", "upsample5_0.main_bn.running_mean", "upsample5_0.main_bn.running_var", "upsample5_0.main_bn.num_batches_tracked", "upsample5_0.ext1_conv.weight", "upsample5_0.ext1_bn.weight", "upsample5_0.ext1_bn.bias", "upsample5_0.ext1_bn.running_mean", "upsample5_0.ext1_bn.running_var", "upsample5_0.ext1_bn.num_batches_tracked", "upsample5_0.ext2_conv.weight", "upsample5_0.ext2_bn.weight", "upsample5_0.ext2_bn.bias", "upsample5_0.ext2_bn.running_mean", "upsample5_0.ext2_bn.running_var", "upsample5_0.ext2_bn.num_batches_tracked", "upsample5_0.ext3_conv.weight", "upsample5_0.ext3_bn.weight", "upsample5_0.ext3_bn.bias", "upsample5_0.ext3_bn.running_mean", "upsample5_0.ext3_bn.running_var", "upsample5_0.ext3_bn.num_batches_tracked", "regular5_1.ext1_conv.weight", "regular5_1.ext1_bn.weight", "regular5_1.ext1_bn.bias", "regular5_1.ext1_bn.running_mean", "regular5_1.ext1_bn.running_var", "regular5_1.ext1_bn.num_batches_tracked", "regular5_1.ext2_conv.weight", "regular5_1.ext2_bn.weight", "regular5_1.ext2_bn.bias", "regular5_1.ext2_bn.running_mean", "regular5_1.ext2_bn.running_var", "regular5_1.ext2_bn.num_batches_tracked", "regular5_1.ext3_conv.weight", "regular5_1.ext3_bn.weight", "regular5_1.ext3_bn.bias", "regular5_1.ext3_bn.running_mean", "regular5_1.ext3_bn.running_var", "regular5_1.ext3_bn.num_batches_tracked", "transposed_conv.weight"]
        deleted_weights = {}
        for key in to_delete:
            deleted_weights[key] = state_dict['state_dict'][key].clone()
            del state_dict['state_dict'][key]
        #print(state_dict['state_dict'].keys())
        net.load_state_dict(state_dict['state_dict'])
        #net_modified.load_state_dict(state_dict['state_dict'])
        print("====> load weights sucessfully!!!")
    else:
        logging.error('can not find the checkpoint.')
        exit(-1)

    net.eval()
    #print(net)   
    input = torch.randn([1, 3, 512, 1024])

    criterion = Criterion(ignore_index=255, weight=None, use_weight=False, reduce=True)
    loss_fn = criterion.to(device)
    
    if args.quant_mode == 'float':
        quant_model = net
    else:
        ## new api
        ####################################################################################
        quantizer = torch_quantizer(args.quant_mode, net, (input), output_dir=args.quant_dir, device=device)

        print('====>quantization<======')
        quant_model = quantizer.quant_model

    if args.eval:
        print('====> Evaluation mIoU')
        eval_miou(args, quant_model, device)    
        #summary(quant_model, input_size=(3,512,1024))
        # handle quantization result
        if args.quant_mode == 'calib':
            print('====>calibration')
            quantizer.export_quant_config()
        if args.quant_mode == 'test' and args.dump_xmodel:
            print('====> xmodel deployment')
            print(strftime("%H:%M:%S", gmtime()))    
            #deploy_check= True if args.dump_golden_data else False
            #shutil.copy('ENet_modified.py','quantize_result/ENet.py')
            dump_xmodel(args.quant_dir, deploy_check=True)
    else:
        print('====> Prediction for demo')
        demo(args, quant_model, net_last, device, deleted_weights)
    #print(quant_model[:-2])



def eval_miou(args, net, device):
    from code.utils import evaluate
    from code.data_loader import cityscapes as citys
    from torch.utils.data import DataLoader
    import torchvision.transforms as standard_transforms
    # data
    print('====> Bulid Dataset...')

    assert args.dataset in ['cityscapes', 'camvid', 'coco', 'voc']
    val_set = citys.CityScapes(root=args.data_root, quality='fine', mode='val', size=(1024, 512))
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False)

    inputs_all, gts_all, predictions_all = [], [], []
    with torch.no_grad():
        for vi, data in enumerate(val_loader):
            print('Process batch id: {} / {}'.format(vi, len(val_loader)))
            inputs, gts = data
            inputs = inputs.to(device)
            gts = gts.to(device)
            outputs = net(inputs)
            if outputs.size()[2:] != gts.size()[1:]:
                outputs = F.interpolate(outputs, size=gts.size()[1:], mode='bilinear', align_corners=True)           
            predictions = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()
            inputs_all.append(inputs.data.cpu())
            gts_all.append(gts.data.cpu().numpy())
            predictions_all.append(predictions)
            if args.dump_xmodel:
                return
        gts_all = np.concatenate(gts_all)
        predictions_all = np.concatenate(predictions_all)
        acc, acc_cls, ious, mean_iu, fwavacc = evaluate(predictions_all, gts_all, args.num_classes, args.ignore_label)
        print('>>>>>>>>>>>>>>>Evaluation Results:>>>>>>>>>>>>>>>>>>>')
        print('Mean IoU(%): {}'.format(mean_iu * 100))
        print('Per-class Mean IoUs: {}'.format(ious))

def demo(args, net, net_last, device, deleted_weights):
    import cv2, glob
    from PIL import Image
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    # read all the images in the folder
    image_list = glob.glob(args.demo_dir + '*')

    # color pallete
    pallete = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
               0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32 ]

    mean = [.485, .456, .406]
    std =  [.229, .224, .225]

    for i, imgName in enumerate(image_list):
        name = imgName.split('/')[-1]
        img = cv2.imread(imgName).astype(np.float32)
        H, W = img.shape[0], img.shape[1]
        img = cv2.resize(img, (args.input_size[0], args.input_size[1]))
        img =  img / 255.0
        for j in range(3):
            img[:, :, j] -= mean[j]
        for j in range(3):
            img[:, :, j] /= std[j]
        img = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img)
        img_tensor = torch.unsqueeze(img_tensor, 0) 
        with torch.no_grad():
            img_variable = img_tensor.to(device)
            #outputs = net(img_variable)
            features, main = net(img_variable)
            #print(outputs.shape)
            #outputs = get_corrected_data() # from fault correction py file

            checkpoint = torch.load(args.weight, map_location=device)
            state_dict = checkpoint['state_dict'] 
            keys_to_keep = [
                "initial_block.main_conv.weight",
                "initial_block.main_bn.weight",
                "initial_block.main_bn.bias",
                "initial_block.main_bn.running_mean",
                "initial_block.main_bn.running_var",
                "initial_block.main_bn.num_batches_tracked",
                "downsample1_0.main_conv.weight", "downsample1_0.main_bn.weight", "downsample1_0.main_bn.bias", "downsample1_0.main_bn.running_mean", "downsample1_0.main_bn.running_var", "downsample1_0.main_bn.num_batches_tracked", "downsample1_0.ext1_conv.weight", "downsample1_0.ext1_bn.weight", "downsample1_0.ext1_bn.bias", "downsample1_0.ext1_bn.running_mean", "downsample1_0.ext1_bn.running_var", "downsample1_0.ext1_bn.num_batches_tracked", "downsample1_0.ext2_conv.weight", "downsample1_0.ext2_bn.weight", "downsample1_0.ext2_bn.bias", "downsample1_0.ext2_bn.running_mean", "downsample1_0.ext2_bn.running_var", "downsample1_0.ext2_bn.num_batches_tracked", "downsample1_0.ext3_conv.weight", "downsample1_0.ext3_bn.weight", "downsample1_0.ext3_bn.bias", "downsample1_0.ext3_bn.running_mean", "downsample1_0.ext3_bn.running_var", "downsample1_0.ext3_bn.num_batches_tracked", "regular1_1.ext1_conv.weight", "regular1_1.ext1_bn.weight", "regular1_1.ext1_bn.bias", "regular1_1.ext1_bn.running_mean", "regular1_1.ext1_bn.running_var", "regular1_1.ext1_bn.num_batches_tracked", "regular1_1.ext2_conv.weight", "regular1_1.ext2_bn.weight", "regular1_1.ext2_bn.bias", "regular1_1.ext2_bn.running_mean", "regular1_1.ext2_bn.running_var", "regular1_1.ext2_bn.num_batches_tracked", "regular1_1.ext3_conv.weight", "regular1_1.ext3_bn.weight", "regular1_1.ext3_bn.bias", "regular1_1.ext3_bn.running_mean", "regular1_1.ext3_bn.running_var", "regular1_1.ext3_bn.num_batches_tracked", "regular1_2.ext1_conv.weight", "regular1_2.ext1_bn.weight", "regular1_2.ext1_bn.bias", "regular1_2.ext1_bn.running_mean", "regular1_2.ext1_bn.running_var", "regular1_2.ext1_bn.num_batches_tracked", "regular1_2.ext2_conv.weight", "regular1_2.ext2_bn.weight", "regular1_2.ext2_bn.bias", "regular1_2.ext2_bn.running_mean", "regular1_2.ext2_bn.running_var", "regular1_2.ext2_bn.num_batches_tracked", "regular1_2.ext3_conv.weight", "regular1_2.ext3_bn.weight", "regular1_2.ext3_bn.bias", "regular1_2.ext3_bn.running_mean", "regular1_2.ext3_bn.running_var", "regular1_2.ext3_bn.num_batches_tracked", "regular1_3.ext3_conv.weight", "regular1_3.ext3_bn.weight", "regular1_3.ext3_bn.bias", "regular1_3.ext3_bn.running_mean", "regular1_3.ext3_bn.running_var", "regular1_3.ext3_bn.num_batches_tracked", "regular1_4.ext1_conv.weight", "regular1_4.ext1_bn.weight", "regular1_4.ext1_bn.bias", "regular1_4.ext1_bn.running_mean", "regular1_4.ext1_bn.running_var", "regular1_4.ext1_bn.num_batches_tracked", "regular1_4.ext2_conv.weight", "regular1_4.ext2_bn.weight", "regular1_4.ext2_bn.bias", "regular1_4.ext2_bn.running_mean", "regular1_4.ext2_bn.running_var", "regular1_4.ext2_bn.num_batches_tracked", "regular1_4.ext3_conv.weight", "regular1_4.ext3_bn.weight", "regular1_4.ext3_bn.bias", "regular1_4.ext3_bn.running_mean", "regular1_4.ext3_bn.running_var", "regular1_4.ext3_bn.num_batches_tracked", "downsample2_0.main_conv.weight", "downsample2_0.main_bn.weight", "downsample2_0.main_bn.bias", "downsample2_0.main_bn.running_mean", "downsample2_0.main_bn.running_var", "downsample2_0.main_bn.num_batches_tracked", "downsample2_0.ext1_conv.weight", "downsample2_0.ext1_bn.weight", "downsample2_0.ext1_bn.bias", "downsample2_0.ext1_bn.running_mean", "downsample2_0.ext1_bn.running_var", "downsample2_0.ext1_bn.num_batches_tracked", "downsample2_0.ext2_conv.weight", "downsample2_0.ext2_bn.weight", "downsample2_0.ext2_bn.bias", "downsample2_0.ext2_bn.running_mean", "downsample2_0.ext2_bn.running_var", "downsample2_0.ext2_bn.num_batches_tracked", "downsample2_0.ext3_conv.weight", "downsample2_0.ext3_bn.weight", "downsample2_0.ext3_bn.bias", "downsample2_0.ext3_bn.running_mean", "downsample2_0.ext3_bn.running_var", "downsample2_0.ext3_bn.num_batches_tracked", "regular2_1.ext1_conv.weight", "regular2_1.ext1_bn.weight", "regular2_1.ext1_bn.bias", "regular2_1.ext1_bn.running_mean", "regular2_1.ext1_bn.running_var", "regular2_1.ext1_bn.num_batches_tracked", "regular2_1.ext2_conv.weight", "regular2_1.ext2_bn.weight", "regular2_1.ext2_bn.bias", "regular2_1.ext2_bn.running_mean", "regular2_1.ext2_bn.running_var", "regular2_1.ext2_bn.num_batches_tracked", "regular2_1.ext3_conv.weight", "regular2_1.ext3_bn.weight", "regular2_1.ext3_bn.bias", "regular2_1.ext3_bn.running_mean", "regular2_1.ext3_bn.running_var", "regular2_1.ext3_bn.num_batches_tracked", "dilated2_2.ext1_conv.weight", "dilated2_2.ext1_bn.weight", "dilated2_2.ext1_bn.bias", "dilated2_2.ext1_bn.running_mean", "dilated2_2.ext1_bn.running_var", "dilated2_2.ext1_bn.num_batches_tracked", "dilated2_2.ext2_conv.weight", "dilated2_2.ext2_bn.weight", "dilated2_2.ext2_bn.bias", "dilated2_2.ext2_bn.running_mean", "dilated2_2.ext2_bn.running_var", "dilated2_2.ext2_bn.num_batches_tracked", "dilated2_2.ext3_conv.weight", "dilated2_2.ext3_bn.weight", "dilated2_2.ext3_bn.bias", "dilated2_2.ext3_bn.running_mean", "dilated2_2.ext3_bn.running_var", "dilated2_2.ext3_bn.num_batches_tracked", "asymmetric2_3.ext1_conv.weight", "asymmetric2_3.ext1_bn.weight", "asymmetric2_3.ext1_bn.bias", "asymmetric2_3.ext1_bn.running_mean", "asymmetric2_3.ext1_bn.running_var", "asymmetric2_3.ext1_bn.num_batches_tracked", "asymmetric2_3.ext2_conv.weight", "asymmetric2_3.ext2_bn.weight", "asymmetric2_3.ext2_bn.bias", "asymmetric2_3.ext2_bn.running_mean", "asymmetric2_3.ext2_bn.running_var", "asymmetric2_3.ext2_bn.num_batches_tracked", "asymmetric2_3.ext3_conv.weight", "asymmetric2_3.ext3_bn.weight", "asymmetric2_3.ext3_bn.bias", "asymmetric2_3.ext3_bn.running_mean", "asymmetric2_3.ext3_bn.running_var", "asymmetric2_3.ext3_bn.num_batches_tracked", "dilated2_4.ext1_conv.weight", "dilated2_4.ext1_bn.weight", "dilated2_4.ext1_bn.bias", "dilated2_4.ext1_bn.running_mean", "dilated2_4.ext1_bn.running_var", "dilated2_4.ext1_bn.num_batches_tracked", "dilated2_4.ext2_conv.weight", "dilated2_4.ext2_bn.weight", "dilated2_4.ext2_bn.bias", "dilated2_4.ext2_bn.running_mean", "dilated2_4.ext2_bn.running_var", "dilated2_4.ext2_bn.num_batches_tracked", "dilated2_4.ext3_conv.weight", "dilated2_4.ext3_bn.weight", "dilated2_4.ext3_bn.bias", "dilated2_4.ext3_bn.running_mean", "dilated2_4.ext3_bn.running_var", "dilated2_4.ext3_bn.num_batches_tracked", "regular2_5.ext1_conv.weight", "regular2_5.ext1_bn.weight", "regular2_5.ext1_bn.bias", "regular2_5.ext1_bn.running_mean", "regular2_5.ext1_bn.running_var", "regular2_5.ext1_bn.num_batches_tracked", "regular2_5.ext2_conv.weight", "regular2_5.ext2_bn.weight", "regular2_5.ext2_bn.bias", "regular2_5.ext2_bn.running_mean", "regular2_5.ext2_bn.running_var", "regular2_5.ext2_bn.num_batches_tracked", "regular2_5.ext3_conv.weight", "regular2_5.ext3_bn.weight", "regular2_5.ext3_bn.bias", "regular2_5.ext3_bn.running_mean", "regular2_5.ext3_bn.running_var", "regular2_5.ext3_bn.num_batches_tracked", "dilated2_6.ext1_conv.weight", "dilated2_6.ext1_bn.weight", "dilated2_6.ext1_bn.bias", "dilated2_6.ext1_bn.running_mean", "dilated2_6.ext1_bn.running_var", "dilated2_6.ext1_bn.num_batches_tracked", "dilated2_6.ext2_conv.weight", "dilated2_6.ext2_bn.weight", "dilated2_6.ext2_bn.bias", "dilated2_6.ext2_bn.running_mean", "dilated2_6.ext2_bn.running_var", "dilated2_6.ext2_bn.num_batches_tracked", "dilated2_6.ext3_conv.weight", "dilated2_6.ext3_bn.weight", "dilated2_6.ext3_bn.bias", "dilated2_6.ext3_bn.running_mean", "dilated2_6.ext3_bn.running_var", "dilated2_6.ext3_bn.num_batches_tracked", "asymmetric2_7.ext1_conv.weight", "asymmetric2_7.ext1_bn.weight", "asymmetric2_7.ext1_bn.bias", "asymmetric2_7.ext1_bn.running_mean", "asymmetric2_7.ext1_bn.running_var", "asymmetric2_7.ext1_bn.num_batches_tracked", "asymmetric2_7.ext2_conv.weight", "asymmetric2_7.ext2_bn.weight", "asymmetric2_7.ext2_bn.bias", "asymmetric2_7.ext2_bn.running_mean", "asymmetric2_7.ext2_bn.running_var", "asymmetric2_7.ext2_bn.num_batches_tracked", "asymmetric2_7.ext3_conv.weight", "asymmetric2_7.ext3_bn.weight", "asymmetric2_7.ext3_bn.bias", "asymmetric2_7.ext3_bn.running_mean", "asymmetric2_7.ext3_bn.running_var", "asymmetric2_7.ext3_bn.num_batches_tracked", "dilated2_8.ext1_conv.weight", "dilated2_8.ext1_bn.weight", "dilated2_8.ext1_bn.bias", "dilated2_8.ext1_bn.running_mean", "dilated2_8.ext1_bn.running_var", "dilated2_8.ext1_bn.num_batches_tracked", "dilated2_8.ext2_conv.weight", "dilated2_8.ext2_bn.weight", "dilated2_8.ext2_bn.bias", "dilated2_8.ext2_bn.running_mean", "dilated2_8.ext2_bn.running_var", "dilated2_8.ext2_bn.num_batches_tracked", "dilated2_8.ext3_conv.weight", "dilated2_8.ext3_bn.weight", "dilated2_8.ext3_bn.bias", "dilated2_8.ext3_bn.running_mean", "dilated2_8.ext3_bn.running_var", "dilated2_8.ext3_bn.num_batches_tracked", "regular3_0.ext1_conv.weight", "regular3_0.ext1_bn.weight", "regular3_0.ext1_bn.bias", "regular3_0.ext1_bn.running_mean", "regular3_0.ext1_bn.running_var", "regular3_0.ext1_bn.num_batches_tracked", "regular3_0.ext2_conv.weight", "regular3_0.ext2_bn.weight", "regular3_0.ext2_bn.bias", "regular3_0.ext2_bn.running_mean", "regular3_0.ext2_bn.running_var", "regular3_0.ext2_bn.num_batches_tracked", "regular3_0.ext3_conv.weight", "regular3_0.ext3_bn.weight", "regular3_0.ext3_bn.bias", "regular3_0.ext3_bn.running_mean", "regular3_0.ext3_bn.running_var", "regular3_0.ext3_bn.num_batches_tracked", "dilated3_1.ext1_conv.weight", "dilated3_1.ext1_bn.weight", "dilated3_1.ext1_bn.bias", "dilated3_1.ext1_bn.running_mean", "dilated3_1.ext1_bn.running_var", "dilated3_1.ext1_bn.num_batches_tracked", "dilated3_1.ext2_conv.weight", "dilated3_1.ext2_bn.weight", "dilated3_1.ext2_bn.bias", "dilated3_1.ext2_bn.running_mean", "dilated3_1.ext2_bn.running_var", "dilated3_1.ext2_bn.num_batches_tracked", "dilated3_1.ext3_conv.weight", "dilated3_1.ext3_bn.weight", "dilated3_1.ext3_bn.bias", "dilated3_1.ext3_bn.running_mean", "dilated3_1.ext3_bn.running_var", "dilated3_1.ext3_bn.num_batches_tracked", "asymmetric3_2.ext1_conv.weight", "asymmetric3_2.ext1_bn.weight", "asymmetric3_2.ext1_bn.bias", "asymmetric3_2.ext1_bn.running_mean", "asymmetric3_2.ext1_bn.running_var", "asymmetric3_2.ext1_bn.num_batches_tracked", "asymmetric3_2.ext2_conv.weight", "asymmetric3_2.ext2_bn.weight", "asymmetric3_2.ext2_bn.bias", "asymmetric3_2.ext2_bn.running_mean", "asymmetric3_2.ext2_bn.running_var", "asymmetric3_2.ext2_bn.num_batches_tracked", "asymmetric3_2.ext3_conv.weight", "asymmetric3_2.ext3_bn.weight", "asymmetric3_2.ext3_bn.bias", "asymmetric3_2.ext3_bn.running_mean", "asymmetric3_2.ext3_bn.running_var", "asymmetric3_2.ext3_bn.num_batches_tracked", "dilated3_3.ext1_conv.weight", "dilated3_3.ext1_bn.weight", "dilated3_3.ext1_bn.bias", "dilated3_3.ext1_bn.running_mean", "dilated3_3.ext1_bn.running_var", "dilated3_3.ext1_bn.num_batches_tracked", "dilated3_3.ext2_conv.weight", "dilated3_3.ext2_bn.weight", "dilated3_3.ext2_bn.bias", "dilated3_3.ext2_bn.running_mean", "dilated3_3.ext2_bn.running_var", "dilated3_3.ext2_bn.num_batches_tracked", "dilated3_3.ext3_conv.weight", "dilated3_3.ext3_bn.weight", "dilated3_3.ext3_bn.bias", "dilated3_3.ext3_bn.running_mean", "dilated3_3.ext3_bn.running_var", "dilated3_3.ext3_bn.num_batches_tracked", "regular3_4.ext1_conv.weight", "regular3_4.ext1_bn.weight", "regular3_4.ext1_bn.bias", "regular3_4.ext1_bn.running_mean", "regular3_4.ext1_bn.running_var", "regular3_4.ext1_bn.num_batches_tracked", "regular3_4.ext2_conv.weight", "regular3_4.ext2_bn.weight", "regular3_4.ext2_bn.bias", "regular3_4.ext2_bn.running_mean", "regular3_4.ext2_bn.running_var", "regular3_4.ext2_bn.num_batches_tracked", "regular3_4.ext3_conv.weight", "regular3_4.ext3_bn.weight", "regular3_4.ext3_bn.bias", "regular3_4.ext3_bn.running_mean", "regular3_4.ext3_bn.running_var", "regular3_4.ext3_bn.num_batches_tracked", "dilated3_5.ext1_conv.weight", "dilated3_5.ext1_bn.weight", "dilated3_5.ext1_bn.bias", "dilated3_5.ext1_bn.running_mean", "dilated3_5.ext1_bn.running_var", "dilated3_5.ext1_bn.num_batches_tracked", "dilated3_5.ext2_conv.weight", "dilated3_5.ext2_bn.weight", "dilated3_5.ext2_bn.bias", "dilated3_5.ext2_bn.running_mean", "dilated3_5.ext2_bn.running_var", "dilated3_5.ext2_bn.num_batches_tracked", "dilated3_5.ext3_conv.weight", "dilated3_5.ext3_bn.weight", "dilated3_5.ext3_bn.bias", "dilated3_5.ext3_bn.running_mean", "dilated3_5.ext3_bn.running_var", "dilated3_5.ext3_bn.num_batches_tracked", "asymmetric3_6.ext1_conv.weight", "asymmetric3_6.ext1_bn.weight", "asymmetric3_6.ext1_bn.bias", "asymmetric3_6.ext1_bn.running_mean", "asymmetric3_6.ext1_bn.running_var", "asymmetric3_6.ext1_bn.num_batches_tracked", "asymmetric3_6.ext2_conv.weight", "asymmetric3_6.ext2_bn.weight", "asymmetric3_6.ext2_bn.bias", "asymmetric3_6.ext2_bn.running_mean", "asymmetric3_6.ext2_bn.running_var", "asymmetric3_6.ext2_bn.num_batches_tracked", "asymmetric3_6.ext3_conv.weight", "asymmetric3_6.ext3_bn.weight", "asymmetric3_6.ext3_bn.bias", "asymmetric3_6.ext3_bn.running_mean", "asymmetric3_6.ext3_bn.running_var", "asymmetric3_6.ext3_bn.num_batches_tracked", "dilated3_7.ext1_conv.weight", "dilated3_7.ext1_bn.weight", "dilated3_7.ext1_bn.bias", "dilated3_7.ext1_bn.running_mean", "dilated3_7.ext1_bn.running_var", "dilated3_7.ext1_bn.num_batches_tracked", "dilated3_7.ext2_conv.weight", "dilated3_7.ext2_bn.weight", "dilated3_7.ext2_bn.bias", "dilated3_7.ext2_bn.running_mean", "dilated3_7.ext2_bn.running_var", "dilated3_7.ext2_bn.num_batches_tracked", "dilated3_7.ext3_conv.weight", "dilated3_7.ext3_bn.weight", "dilated3_7.ext3_bn.bias", "dilated3_7.ext3_bn.running_mean", "dilated3_7.ext3_bn.running_var", "dilated3_7.ext3_bn.num_batches_tracked"
            ]
            keys_to_delete = ["upsample4_0.main_conv.weight", "upsample4_0.main_bn.weight", "upsample4_0.main_bn.bias", "upsample4_0.main_bn.running_mean", "upsample4_0.main_bn.running_var", "upsample4_0.main_bn.num_batches_tracked"]
            keep_dict = {}
            for key in keys_to_keep:
                keep_dict[key] = checkpoint['state_dict'][key].clone()
            for key in list(state_dict.keys()):
                if key in deleted_weights or key in keep_dict:
                    continue
                else:
                    del state_dict[key]
            #print(checkpoint['state_dict'].keys())
            deleted_weights = {}
            for key in keys_to_delete:
                deleted_weights[key] = checkpoint['state_dict'][key].clone()
                del checkpoint['state_dict'][key]

            net_last.load_state_dict(checkpoint['state_dict'])

            img_name = "enet_inner_layer_0_90128_90129_200_1"
            path = "data/demo_results/enet_stages/stage_4_upsample_intralayer1/" + img_name + ".csv"
            accuracies = []

            #args.save_dir = "data/demo_results/enet_stages/stage_2/"
            csv_files = glob.glob(os.path.join(args.save_dir, "*.csv"))
            
            #for path in csv_files:
            #print(f"Processing {path[37:]}...")
            corrected = net_last(get_corrected_data(path), main)
            faulty = net_last(get_faulty_data(path), main)
            free = net_last(get_free_data(path), main) 
            # z1 = net_last(get_free_data(path))[0].max(0)[1].byte()v.cpu().data.numpy().flatten()
            # z2 = net_last(get_faulty_data(path))[0].max(0)[1].byte().cpu().data.numpy().flatten()
            # z3 = corrected[0].max(0)[1].byte().cpu().data.numpy().flatten()
            #np.savetxt("data/demo_results/tensor_output.csv", [z1, z2, z3], delimiter=",", fmt='%d')

            if corrected.size()[-1] != W:
                corrected = F.interpolate(corrected, size=(H, W), mode='bilinear', align_corners=True)
            classMap_numpy_fixed = corrected[0].max(0)[1].byte().cpu().data.numpy()
            classMap_numpy_fixed = Image.fromarray(classMap_numpy_fixed)
            name = imgName.split('/')[-1]
            classMap_numpy_color = classMap_numpy_fixed.copy()
            classMap_numpy_color.putpalette(pallete)
            classMap_numpy_color.save(os.path.join(args.save_dir, 'CORRECTEDcolor_'+ img_name +".png"))
            #classMap_numpy_color.save(os.path.join(args.save_dir, path[38:] + '_color_CORRECTED.png'))
            
            if faulty.size()[-1] != W:
                faulty = F.interpolate(faulty, size=(H, W), mode='bilinear', align_corners=True)
            classMap_numpy_fixed = faulty[0].max(0)[1].byte().cpu().data.numpy()
            classMap_numpy_fixed = Image.fromarray(classMap_numpy_fixed)
            name = imgName.split('/')[-1]
            classMap_numpy_color = classMap_numpy_fixed.copy()
            classMap_numpy_color.putpalette(pallete)
            classMap_numpy_color.save(os.path.join(args.save_dir, 'FAULTYcolor_'+ img_name +".png"))
            #classMap_numpy_color.save(os.path.join(args.save_dir, path[38:] + '_color_FAULTY.png'))

            if free.size()[-1] != W:
                free = F.interpolate(free, size=(H, W), mode='bilinear', align_corners=True)
            classMap_numpy_fixed = free[0].max(0)[1].byte().cpu().data.numpy()
            classMap_numpy_fixed = Image.fromarray(classMap_numpy_fixed)
            name = imgName.split('/')[-1]
            classMap_numpy_color = classMap_numpy_fixed.copy()
            classMap_numpy_color.putpalette(pallete)
            classMap_numpy_color.save(os.path.join(args.save_dir, 'FREEcolor_'+ img_name+".png"))



if __name__ == '__main__':
    args = Configs().parse()
    main(args)






#FOR DEMO
# path = "data/demo_results/img_analysis/width_200/img0/enet_inner_layer_0_90154_90155_200_1.csv"
            # free = get_free_data(path)
            # #quantizer = torch_quantizer(args.quant_mode, net_last, (free,), output_dir=args.quant_dir, device=device)
            # free = net_last(free) # include the last layer
            # if free.size()[-1] != W:
            #     free = F.interpolate(free, size=(H, W), mode='bilinear', align_corners=True)
            # classMap_numpy_free = free[0].max(0)[1].byte().cpu().data.numpy()
            # faulty = get_faulty_data(path)
            # faulty = net_last(faulty) # include the last layer
            # if faulty.size()[-1] != W:
            #     faulty = F.interpolate(faulty, size=(H, W), mode='bilinear', align_corners=True)
            # classMap_numpy_faulty = faulty[0].max(0)[1].byte().cpu().data.numpy()
            # for threshold in range(0, 90):
            #     fixed = get_corrected_data(path, 19)
            #     fixed = net_last(fixed) # include the last layer
            #     if fixed.size()[-1] != W:
            #         fixed = F.interpolate(fixed, size=(H, W), mode='bilinear', align_corners=True)
            #     classMap_numpy_fixed = fixed[0].max(0)[1].byte().cpu().data.numpy()

            #     # Flatten all arrays
            #     flat_free   = classMap_numpy_free.flatten()
            #     flat_fixed  = classMap_numpy_fixed.flatten()
            #     flat_faulty = classMap_numpy_faulty.flatten()
            #     corrected_indices = np.where(flat_fixed != flat_faulty)[0]
            #     if corrected_indices.size == 0:
            #         print("⚠️ No values were corrected.")
            #     else:
            #         fixed_vals = flat_fixed[corrected_indices]
            #         free_vals  = flat_free[corrected_indices]
            #         correct = np.sum(fixed_vals == free_vals)
            #         total = len(corrected_indices)
            #         accuracy = correct / total * 100
            #         print(f"{total} values were corrected.")
            #         print(f"Correct corrections (matched ground truth): {correct}")
            #         print(f"Correction Accuracy: {accuracy:.2f}%")

# if free.size()[-1] != W:
            #     free = F.interpolate(free, size=(H, W), mode='bilinear', align_corners=True)
            # classMap_numpy_free = free[0].max(0)[1].byte().cpu().data.numpy()

            # faulty = get_faulty_data(path)
            # faulty = net_last(faulty)  # include the last layer
            # if faulty.size()[-1] != W:
            #     faulty = F.interpolate(faulty, size=(H, W), mode='bilinear', align_corners=True)
            # classMap_numpy_faulty = faulty[0].max(0)[1].byte().cpu().data.numpy()
            # num_corrected_list = []


            # for threshold in range(0, 126):
            #     fixed = get_corrected_data(path, threshold)  # ← Use threshold here
            #     fixed = net_last(fixed)  # include the last layer
            #     if fixed.size()[-1] != W:
            #         fixed = F.interpolate(fixed, size=(H, W), mode='bilinear', align_corners=True)
            #     classMap_numpy_fixed = fixed[0].max(0)[1].byte().cpu().data.numpy()
            #
            #     # Flatten all arrays
            #     flat_free   = classMap_numpy_free.flatten()
            #     flat_fixed  = classMap_numpy_fixed.flatten()
            #     flat_faulty = classMap_numpy_faulty.flatten()
            #     corrected_indices = np.where(flat_fixed != flat_faulty)[0]
            #     num_corrected = len(corrected_indices)
            #     num_corrected_list.append(num_corrected)
            #     if corrected_indices.size == 0:
            #         print(f"Threshold {threshold}: ⚠️ No values were corrected.")
            #         accuracies.append(0)
            #     else:
            #         fixed_vals = flat_fixed[corrected_indices]
            #         free_vals  = flat_free[corrected_indices]
            #         correct = np.sum(fixed_vals == free_vals)
            #         total = len(corrected_indices)
            #         accuracy = correct / total * 100
            #         print(f"Threshold {threshold}: {total} corrected | {correct} matched | Accuracy = {accuracy:.2f}%")
            #         accuracies.append(accuracy)
            # print(accuracies)
            # print(num_corrected_list)