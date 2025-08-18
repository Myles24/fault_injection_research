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


# MIT License

# Copyright (c) 2019 Hengshuang Zhao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



import os
import sys
if os.environ["W_QUANT"]=='1':
    # load quant apis
    from pytorch_nndct.apis import torch_quantizer, dump_xmodel

import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transform
import torch.nn.functional as F
from PIL import Image
import argparse
import logging
import glob
from torchsummary import summary
#from code.configs.model_config import Options
from fault_correction import get_free_data, get_faulty_data, get_corrected_data

class Configs():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch SemanticFPN model')
        # model and dataset 
        parser.add_argument('--model', type=str, default='fpn', help='model name (default: fpn)')
        parser.add_argument('--backbone', type=str, default='mobilenetv2',choices=['resnet18', 'mobilenetv2'], \
                             help='backbone name (default: resnet18)')
        parser.add_argument('--dataset', type=str, default='citys',help='dataset name (default: cityscapes)')
        parser.add_argument('--num-classes', type=int, default=19, help='the classes numbers (default: 19 for cityscapes)')
        parser.add_argument('--data-folder', type=str, default='./data/cityscapes',help='training dataset folder (default: ./data)')
        parser.add_argument('--ignore_label', type=int, default=-1, help='the ignore label (default: 255 for cityscapes)')

        parser.add_argument('--base-size', type=int, default=1024, help='the shortest image size')
        parser.add_argument('--crop-size', type=int, default=512, help='input size for inference')
        parser.add_argument('--batch-size', type=int, default=1,metavar='N', help='input batch size for testing (default: 10)')
        # cuda, seed and logging
        parser.add_argument('--workers', type=int, default=16, metavar='N', help='dataloader threads')
        parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
        # checking point
        parser.add_argument('--weight', type=str, default=None, help='path to final weight')
        # evaluation option
        parser.add_argument('--eval', action='store_true', default=False, help='evaluating mIoU')
        # test option
        parser.add_argument('--scale', type=float, default=0.5, help='downsample scale')
        parser.add_argument('--test-folder', type=str, default=None, help='path to demo folder')
        parser.add_argument('--save-dir', type=str, default='./data/demo_results')

        #quantization options
        parser.add_argument('--quant_dir', type=str, default='quantize_result', help='path to save quant info')
        parser.add_argument('--quant_mode', default='calib', choices=['float', 'calib', 'test'], \
           help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
        parser.add_argument('--fast_finetune', dest='fast_finetune', action='store_true', help='fast finetune model before calibration')
        parser.add_argument('--finetune', dest='finetune', action='store_true', help='finetune model before calibration')
        parser.add_argument('--dump_xmodel', dest='dump_xmodel', action='store_true', help='dump xmodel after test')
        parser.add_argument('--device', default='cpu', choices=['gpu', 'cpu'], help='assign runtime device')

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args

class Criterion(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, weight=None, use_weight=True, reduce=True):
        super(Criterion, self).__init__()
        #class_wts = torch.ones(len(weight))
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.criterion(scale_pred, target)
        return loss

def build_data(args, subset_len=None, sample_method='random'):
    from code.datasets import get_segmentation_dataset
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])

    data_kwargs = {'transform': input_transform, 'base_size': args.base_size,'crop_size': args.crop_size}
    if args.eval:
        testset = get_segmentation_dataset(args.dataset, split='val', mode='testval', root=args.data_folder,**data_kwargs)
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
    if subset_len:
        assert subset_len <= len(testset)
        if sample_method == 'random':
            testset = torch.utils.data.Subset(testset, random.sample(range(0, len(test_data)), subset_len))
        else:
            testset = torch.utils.data.Subset(testset, list(range(subset_len)))
    #dataloader 
    test_data = data.DataLoader(testset, batch_size=args.batch_size,drop_last=False, shuffle=False)
    return test_data

''' 
def build_data(args):
    from code.datasets import get_segmentation_dataset
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])

    data_kwargs = {'transform': input_transform, 'base_size': args.base_size,'crop_size': args.crop_size}
    if args.eval:
        testset = get_segmentation_dataset(args.dataset, split='val', mode='testval', root=args.data_folder,**data_kwargs)
    # dataloader
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
    test_data = data.DataLoader(testset, batch_size=args.batch_size,drop_last=False, shuffle=False)
    return test_data
'''

def build_model(args, device):
    from code.models import fpn
    model = fpn.get_fpn(nclass=args.num_classes, backbone=args.backbone, pretrained=False).to(device)
    checkpoint = torch.load(args.weight, map_location=device)
    
    checkpoint['state_dict'] = OrderedDict([(k[5:], v) if 'base' in k else (k, v) for k, v in checkpoint['state_dict'].items()])
    #checkpoint['state_dict']['module_119.bias'] = checkpoint['state_dict']['head.up2.conv_out.conv1.weight']
    #print(checkpoint['state_dict'].keys())
    to_delete= ['head.up1.down_conv.weight', 'head.up1.down_bn.weight', 'head.up1.down_bn.bias', 'head.up1.down_bn.running_mean', 'head.up1.down_bn.running_var', 'head.up1.down_bn.num_batches_tracked', 'head.up1.conv_enc.conv1.weight', 'head.up1.conv_enc.bn1.weight', 'head.up1.conv_enc.bn1.bias', 'head.up1.conv_enc.bn1.running_mean', 'head.up1.conv_enc.bn1.running_var', 'head.up1.conv_enc.bn1.num_batches_tracked', 'head.up1.conv_enc.conv2.weight', 'head.up1.conv_enc.bn2.weight', 'head.up1.conv_enc.bn2.bias', 'head.up1.conv_enc.bn2.running_mean', 'head.up1.conv_enc.bn2.running_var', 'head.up1.conv_enc.bn2.num_batches_tracked', 'head.up1.conv_enc.downsample.0.weight', 'head.up1.conv_enc.downsample.1.weight', 'head.up1.conv_enc.downsample.1.bias', 'head.up1.conv_enc.downsample.1.running_mean', 'head.up1.conv_enc.downsample.1.running_var', 'head.up1.conv_enc.downsample.1.num_batches_tracked', 'head.up1.conv_out.conv1.weight', 'head.up1.conv_out.bn1.weight', 'head.up1.conv_out.bn1.bias', 'head.up1.conv_out.bn1.running_mean', 'head.up1.conv_out.bn1.running_var', 'head.up1.conv_out.bn1.num_batches_tracked', 'head.up1.conv_up.weight', 'head.conv_up0.weight', 'head.conv_up1.weight']
    for key in to_delete:
        del checkpoint['state_dict'][key]
    model.load_state_dict(checkpoint['state_dict'])
    return model

def build_model2(args, device):
    from code.models import fpn
    model = fpn.get_fpn_last(nclass=args.num_classes, backbone=args.backbone, pretrained=False).to(device)
    checkpoint = torch.load(args.weight, map_location=device)
    #print(checkpoint['state_dict'].keys())
    checkpoint['state_dict'] = OrderedDict([(k[5:], v) if 'base' in k else (k, v) for k, v in checkpoint['state_dict'].items()])
    to_delete= ['head.up1.down_conv.weight', 'head.up1.down_bn.weight', 'head.up1.down_bn.bias', 'head.up1.down_bn.running_mean', 'head.up1.down_bn.running_var', 'head.up1.down_bn.num_batches_tracked', 'head.up1.conv_enc.conv1.weight', 'head.up1.conv_enc.bn1.weight', 'head.up1.conv_enc.bn1.bias', 'head.up1.conv_enc.bn1.running_mean', 'head.up1.conv_enc.bn1.running_var', 'head.up1.conv_enc.bn1.num_batches_tracked', 'head.up1.conv_enc.conv2.weight', 'head.up1.conv_enc.bn2.weight', 'head.up1.conv_enc.bn2.bias', 'head.up1.conv_enc.bn2.running_mean', 'head.up1.conv_enc.bn2.running_var', 'head.up1.conv_enc.bn2.num_batches_tracked', 'head.up1.conv_enc.downsample.0.weight', 'head.up1.conv_enc.downsample.1.weight', 'head.up1.conv_enc.downsample.1.bias', 'head.up1.conv_enc.downsample.1.running_mean', 'head.up1.conv_enc.downsample.1.running_var', 'head.up1.conv_enc.downsample.1.num_batches_tracked', 'head.up1.conv_out.conv1.weight', 'head.up1.conv_out.bn1.weight', 'head.up1.conv_out.bn1.bias', 'head.up1.conv_out.bn1.running_mean', 'head.up1.conv_out.bn1.running_var', 'head.up1.conv_out.bn1.num_batches_tracked', 'head.up1.conv_up.weight', 'head.conv_up0.weight', 'head.conv_up1.weight']
    to_delete = ["pretrained.features.0.0.weight", "pretrained.features.0.1.weight", "pretrained.features.0.1.bias", "pretrained.features.0.1.running_mean", "pretrained.features.0.1.running_var", "pretrained.features.0.1.num_batches_tracked", "pretrained.features.1.conv.0.0.weight", "pretrained.features.1.conv.0.1.weight", "pretrained.features.1.conv.0.1.bias", "pretrained.features.1.conv.0.1.running_mean", "pretrained.features.1.conv.0.1.running_var", "pretrained.features.1.conv.0.1.num_batches_tracked", "pretrained.features.1.conv.1.weight", "pretrained.features.1.conv.2.weight", "pretrained.features.1.conv.2.bias", "pretrained.features.1.conv.2.running_mean", "pretrained.features.1.conv.2.running_var", "pretrained.features.1.conv.2.num_batches_tracked", "pretrained.features.2.conv.0.0.weight", "pretrained.features.2.conv.0.1.weight", "pretrained.features.2.conv.0.1.bias", "pretrained.features.2.conv.0.1.running_mean", "pretrained.features.2.conv.0.1.running_var", "pretrained.features.2.conv.0.1.num_batches_tracked", "pretrained.features.2.conv.1.0.weight", "pretrained.features.2.conv.1.1.weight", "pretrained.features.2.conv.1.1.bias", "pretrained.features.2.conv.1.1.running_mean", "pretrained.features.2.conv.1.1.running_var", "pretrained.features.2.conv.1.1.num_batches_tracked", "pretrained.features.2.conv.2.weight", "pretrained.features.2.conv.3.weight", "pretrained.features.2.conv.3.bias", "pretrained.features.2.conv.3.running_mean", "pretrained.features.2.conv.3.running_var", "pretrained.features.2.conv.3.num_batches_tracked", "pretrained.features.3.conv.0.0.weight", "pretrained.features.3.conv.0.1.weight", "pretrained.features.3.conv.0.1.bias", "pretrained.features.3.conv.0.1.running_mean", "pretrained.features.3.conv.0.1.running_var", "pretrained.features.3.conv.0.1.num_batches_tracked", "pretrained.features.3.conv.1.0.weight", "pretrained.features.3.conv.1.1.weight", "pretrained.features.3.conv.1.1.bias", "pretrained.features.3.conv.1.1.running_mean", "pretrained.features.3.conv.1.1.running_var", "pretrained.features.3.conv.1.1.num_batches_tracked", "pretrained.features.3.conv.2.weight", "pretrained.features.3.conv.3.weight", "pretrained.features.3.conv.3.bias", "pretrained.features.3.conv.3.running_mean", "pretrained.features.3.conv.3.running_var", "pretrained.features.3.conv.3.num_batches_tracked", "pretrained.features.4.conv.0.0.weight", "pretrained.features.4.conv.0.1.weight", "pretrained.features.4.conv.0.1.bias", "pretrained.features.4.conv.0.1.running_mean", "pretrained.features.4.conv.0.1.running_var", "pretrained.features.4.conv.0.1.num_batches_tracked", "pretrained.features.4.conv.1.0.weight", "pretrained.features.4.conv.1.1.weight", "pretrained.features.4.conv.1.1.bias", "pretrained.features.4.conv.1.1.running_mean", "pretrained.features.4.conv.1.1.running_var", "pretrained.features.4.conv.1.1.num_batches_tracked", "pretrained.features.4.conv.2.weight", "pretrained.features.4.conv.3.weight", "pretrained.features.4.conv.3.bias", "pretrained.features.4.conv.3.running_mean", "pretrained.features.4.conv.3.running_var", "pretrained.features.4.conv.3.num_batches_tracked", "pretrained.features.5.conv.0.0.weight", "pretrained.features.5.conv.0.1.weight", "pretrained.features.5.conv.0.1.bias", "pretrained.features.5.conv.0.1.running_mean", "pretrained.features.5.conv.0.1.running_var", "pretrained.features.5.conv.0.1.num_batches_tracked", "pretrained.features.5.conv.1.0.weight", "pretrained.features.5.conv.1.1.weight", "pretrained.features.5.conv.1.1.bias", "pretrained.features.5.conv.1.1.running_mean", "pretrained.features.5.conv.1.1.running_var", "pretrained.features.5.conv.1.1.num_batches_tracked", "pretrained.features.5.conv.2.weight", "pretrained.features.5.conv.3.weight", "pretrained.features.5.conv.3.bias", "pretrained.features.5.conv.3.running_mean", "pretrained.features.5.conv.3.running_var", "pretrained.features.5.conv.3.num_batches_tracked", "pretrained.features.6.conv.0.0.weight", "pretrained.features.6.conv.0.1.weight", "pretrained.features.6.conv.0.1.bias", "pretrained.features.6.conv.0.1.running_mean", "pretrained.features.6.conv.0.1.running_var", "pretrained.features.6.conv.0.1.num_batches_tracked", "pretrained.features.6.conv.1.0.weight", "pretrained.features.6.conv.1.1.weight", "pretrained.features.6.conv.1.1.bias", "pretrained.features.6.conv.1.1.running_mean", "pretrained.features.6.conv.1.1.running_var", "pretrained.features.6.conv.1.1.num_batches_tracked", "pretrained.features.6.conv.2.weight", "pretrained.features.6.conv.3.weight", "pretrained.features.6.conv.3.bias", "pretrained.features.6.conv.3.running_mean", "pretrained.features.6.conv.3.running_var", "pretrained.features.6.conv.3.num_batches_tracked", "pretrained.features.7.conv.0.0.weight", "pretrained.features.7.conv.0.1.weight", "pretrained.features.7.conv.0.1.bias", "pretrained.features.7.conv.0.1.running_mean", "pretrained.features.7.conv.0.1.running_var", "pretrained.features.7.conv.0.1.num_batches_tracked", "pretrained.features.7.conv.1.0.weight", "pretrained.features.7.conv.1.1.weight", "pretrained.features.7.conv.1.1.bias", "pretrained.features.7.conv.1.1.running_mean", "pretrained.features.7.conv.1.1.running_var", "pretrained.features.7.conv.1.1.num_batches_tracked", "pretrained.features.7.conv.2.weight", "pretrained.features.7.conv.3.weight", "pretrained.features.7.conv.3.bias", "pretrained.features.7.conv.3.running_mean", "pretrained.features.7.conv.3.running_var", "pretrained.features.7.conv.3.num_batches_tracked", "pretrained.features.8.conv.0.0.weight", "pretrained.features.8.conv.0.1.weight", "pretrained.features.8.conv.0.1.bias", "pretrained.features.8.conv.0.1.running_mean", "pretrained.features.8.conv.0.1.running_var", "pretrained.features.8.conv.0.1.num_batches_tracked", "pretrained.features.8.conv.1.0.weight", "pretrained.features.8.conv.1.1.weight", "pretrained.features.8.conv.1.1.bias", "pretrained.features.8.conv.1.1.running_mean", "pretrained.features.8.conv.1.1.running_var", "pretrained.features.8.conv.1.1.num_batches_tracked", "pretrained.features.8.conv.2.weight", "pretrained.features.8.conv.3.weight", "pretrained.features.8.conv.3.bias", "pretrained.features.8.conv.3.running_mean", "pretrained.features.8.conv.3.running_var", "pretrained.features.8.conv.3.num_batches_tracked", "pretrained.features.9.conv.0.0.weight", "pretrained.features.9.conv.0.1.weight", "pretrained.features.9.conv.0.1.bias", "pretrained.features.9.conv.0.1.running_mean", "pretrained.features.9.conv.0.1.running_var", "pretrained.features.9.conv.0.1.num_batches_tracked", "pretrained.features.9.conv.1.0.weight", "pretrained.features.9.conv.1.1.weight", "pretrained.features.9.conv.1.1.bias", "pretrained.features.9.conv.1.1.running_mean", "pretrained.features.9.conv.1.1.running_var", "pretrained.features.9.conv.1.1.num_batches_tracked", "pretrained.features.9.conv.2.weight", "pretrained.features.9.conv.3.weight", "pretrained.features.9.conv.3.bias", "pretrained.features.9.conv.3.running_mean", "pretrained.features.9.conv.3.running_var", "pretrained.features.9.conv.3.num_batches_tracked", "pretrained.features.10.conv.0.0.weight", "pretrained.features.10.conv.0.1.weight", "pretrained.features.10.conv.0.1.bias", "pretrained.features.10.conv.0.1.running_mean", "pretrained.features.10.conv.0.1.running_var", "pretrained.features.10.conv.0.1.num_batches_tracked", "pretrained.features.10.conv.1.0.weight", "pretrained.features.10.conv.1.1.weight", "pretrained.features.10.conv.1.1.bias", "pretrained.features.10.conv.1.1.running_mean", "pretrained.features.10.conv.1.1.running_var", "pretrained.features.10.conv.1.1.num_batches_tracked", "pretrained.features.10.conv.2.weight", "pretrained.features.10.conv.3.weight", "pretrained.features.10.conv.3.bias", "pretrained.features.10.conv.3.running_mean", "pretrained.features.10.conv.3.running_var", "pretrained.features.10.conv.3.num_batches_tracked", "pretrained.features.11.conv.0.0.weight", "pretrained.features.11.conv.0.1.weight", "pretrained.features.11.conv.0.1.bias", "pretrained.features.11.conv.0.1.running_mean", "pretrained.features.11.conv.0.1.running_var", "pretrained.features.11.conv.0.1.num_batches_tracked", "pretrained.features.11.conv.1.0.weight", "pretrained.features.11.conv.1.1.weight", "pretrained.features.11.conv.1.1.bias", "pretrained.features.11.conv.1.1.running_mean", "pretrained.features.11.conv.1.1.running_var", "pretrained.features.11.conv.1.1.num_batches_tracked", "pretrained.features.11.conv.2.weight", "pretrained.features.11.conv.3.weight", "pretrained.features.11.conv.3.bias", "pretrained.features.11.conv.3.running_mean", "pretrained.features.11.conv.3.running_var", "pretrained.features.11.conv.3.num_batches_tracked", "pretrained.features.12.conv.0.0.weight", "pretrained.features.12.conv.0.1.weight", "pretrained.features.12.conv.0.1.bias", "pretrained.features.12.conv.0.1.running_mean", "pretrained.features.12.conv.0.1.running_var", "pretrained.features.12.conv.0.1.num_batches_tracked", "pretrained.features.12.conv.1.0.weight", "pretrained.features.12.conv.1.1.weight", "pretrained.features.12.conv.1.1.bias", "pretrained.features.12.conv.1.1.running_mean", "pretrained.features.12.conv.1.1.running_var", "pretrained.features.12.conv.1.1.num_batches_tracked", "pretrained.features.12.conv.2.weight", "pretrained.features.12.conv.3.weight", "pretrained.features.12.conv.3.bias", "pretrained.features.12.conv.3.running_mean", "pretrained.features.12.conv.3.running_var", "pretrained.features.12.conv.3.num_batches_tracked", "pretrained.features.13.conv.0.0.weight", "pretrained.features.13.conv.0.1.weight", "pretrained.features.13.conv.0.1.bias", "pretrained.features.13.conv.0.1.running_mean", "pretrained.features.13.conv.0.1.running_var", "pretrained.features.13.conv.0.1.num_batches_tracked", "pretrained.features.13.conv.1.0.weight", "pretrained.features.13.conv.1.1.weight", "pretrained.features.13.conv.1.1.bias", "pretrained.features.13.conv.1.1.running_mean", "pretrained.features.13.conv.1.1.running_var", "pretrained.features.13.conv.1.1.num_batches_tracked", "pretrained.features.13.conv.2.weight", "pretrained.features.13.conv.3.weight", "pretrained.features.13.conv.3.bias", "pretrained.features.13.conv.3.running_mean", "pretrained.features.13.conv.3.running_var", "pretrained.features.13.conv.3.num_batches_tracked", "pretrained.features.14.conv.0.0.weight", "pretrained.features.14.conv.0.1.weight", "pretrained.features.14.conv.0.1.bias", "pretrained.features.14.conv.0.1.running_mean", "pretrained.features.14.conv.0.1.running_var", "pretrained.features.14.conv.0.1.num_batches_tracked", "pretrained.features.14.conv.1.0.weight", "pretrained.features.14.conv.1.1.weight", "pretrained.features.14.conv.1.1.bias", "pretrained.features.14.conv.1.1.running_mean", "pretrained.features.14.conv.1.1.running_var", "pretrained.features.14.conv.1.1.num_batches_tracked", "pretrained.features.14.conv.2.weight", "pretrained.features.14.conv.3.weight", "pretrained.features.14.conv.3.bias", "pretrained.features.14.conv.3.running_mean", "pretrained.features.14.conv.3.running_var", "pretrained.features.14.conv.3.num_batches_tracked", "pretrained.features.15.conv.0.0.weight", "pretrained.features.15.conv.0.1.weight", "pretrained.features.15.conv.0.1.bias", "pretrained.features.15.conv.0.1.running_mean", "pretrained.features.15.conv.0.1.running_var", "pretrained.features.15.conv.0.1.num_batches_tracked", "pretrained.features.15.conv.1.0.weight", "pretrained.features.15.conv.1.1.weight", "pretrained.features.15.conv.1.1.bias", "pretrained.features.15.conv.1.1.running_mean", "pretrained.features.15.conv.1.1.running_var", "pretrained.features.15.conv.1.1.num_batches_tracked", "pretrained.features.15.conv.2.weight", "pretrained.features.15.conv.3.weight", "pretrained.features.15.conv.3.bias", "pretrained.features.15.conv.3.running_mean", "pretrained.features.15.conv.3.running_var", "pretrained.features.15.conv.3.num_batches_tracked", "pretrained.features.16.conv.0.0.weight", "pretrained.features.16.conv.0.1.weight", "pretrained.features.16.conv.0.1.bias", "pretrained.features.16.conv.0.1.running_mean", "pretrained.features.16.conv.0.1.running_var", "pretrained.features.16.conv.0.1.num_batches_tracked", "pretrained.features.16.conv.1.0.weight", "pretrained.features.16.conv.1.1.weight", "pretrained.features.16.conv.1.1.bias", "pretrained.features.16.conv.1.1.running_mean", "pretrained.features.16.conv.1.1.running_var", "pretrained.features.16.conv.1.1.num_batches_tracked", "pretrained.features.16.conv.2.weight", "pretrained.features.16.conv.3.weight", "pretrained.features.16.conv.3.bias", "pretrained.features.16.conv.3.running_mean", "pretrained.features.16.conv.3.running_var", "pretrained.features.16.conv.3.num_batches_tracked", "pretrained.features.17.conv.0.0.weight", "pretrained.features.17.conv.0.1.weight", "pretrained.features.17.conv.0.1.bias", "pretrained.features.17.conv.0.1.running_mean", "pretrained.features.17.conv.0.1.running_var", "pretrained.features.17.conv.0.1.num_batches_tracked", "pretrained.features.17.conv.1.0.weight", "pretrained.features.17.conv.1.1.weight", "pretrained.features.17.conv.1.1.bias", "pretrained.features.17.conv.1.1.running_mean", "pretrained.features.17.conv.1.1.running_var", "pretrained.features.17.conv.1.1.num_batches_tracked", "pretrained.features.17.conv.2.weight", "pretrained.features.17.conv.3.weight", "pretrained.features.17.conv.3.bias", "pretrained.features.17.conv.3.running_mean", "pretrained.features.17.conv.3.running_var", "pretrained.features.17.conv.3.num_batches_tracked", "pretrained.features.18.0.weight", "pretrained.features.18.1.weight", "pretrained.features.18.1.bias", "pretrained.features.18.1.running_mean", "pretrained.features.18.1.running_var", "pretrained.features.18.1.num_batches_tracked", "pretrained.classifier.1.weight", "pretrained.classifier.1.bias", "head.conv_enc2dec.weight", "head.bn_enc2dec.weight", "head.bn_enc2dec.bias", "head.bn_enc2dec.running_mean", "head.bn_enc2dec.running_var", "head.bn_enc2dec.num_batches_tracked", "head.up3.down_conv.weight", "head.up3.down_bn.weight", "head.up3.down_bn.bias", "head.up3.down_bn.running_mean", "head.up3.down_bn.running_var", "head.up3.down_bn.num_batches_tracked", "head.up3.conv_enc.conv1.weight", "head.up3.conv_enc.bn1.weight", "head.up3.conv_enc.bn1.bias", "head.up3.conv_enc.bn1.running_mean", "head.up3.conv_enc.bn1.running_var", "head.up3.conv_enc.bn1.num_batches_tracked", "head.up3.conv_enc.conv2.weight", "head.up3.conv_enc.bn2.weight", "head.up3.conv_enc.bn2.bias", "head.up3.conv_enc.bn2.running_mean", "head.up3.conv_enc.bn2.running_var", "head.up3.conv_enc.bn2.num_batches_tracked", "head.up3.conv_enc.downsample.0.weight", "head.up3.conv_enc.downsample.1.weight", "head.up3.conv_enc.downsample.1.bias", "head.up3.conv_enc.downsample.1.running_mean", "head.up3.conv_enc.downsample.1.running_var", "head.up3.conv_enc.downsample.1.num_batches_tracked", "head.up3.conv_out.conv1.weight", "head.up3.conv_out.bn1.weight", "head.up3.conv_out.bn1.bias", "head.up3.conv_out.bn1.running_mean", "head.up3.conv_out.bn1.running_var", "head.up3.conv_out.bn1.num_batches_tracked", "head.up3.conv_up.weight", "head.up2.down_conv.weight", "head.up2.down_bn.weight", "head.up2.down_bn.bias", "head.up2.down_bn.running_mean", "head.up2.down_bn.running_var", "head.up2.down_bn.num_batches_tracked", "head.up2.conv_enc.conv1.weight", "head.up2.conv_enc.bn1.weight", "head.up2.conv_enc.bn1.bias", "head.up2.conv_enc.bn1.running_mean", "head.up2.conv_enc.bn1.running_var", "head.up2.conv_enc.bn1.num_batches_tracked", "head.up2.conv_enc.conv2.weight", "head.up2.conv_enc.bn2.weight", "head.up2.conv_enc.bn2.bias", "head.up2.conv_enc.bn2.running_mean", "head.up2.conv_enc.bn2.running_var", "head.up2.conv_enc.bn2.num_batches_tracked", "head.up2.conv_enc.downsample.0.weight", "head.up2.conv_enc.downsample.1.weight", "head.up2.conv_enc.downsample.1.bias", "head.up2.conv_enc.downsample.1.running_mean", "head.up2.conv_enc.downsample.1.running_var", "head.up2.conv_enc.downsample.1.num_batches_tracked", "head.up2.conv_out.conv1.weight", "head.up2.conv_out.bn1.weight", "head.up2.conv_out.bn1.bias", "head.up2.conv_out.bn1.running_mean", "head.up2.conv_out.bn1.running_var", "head.up2.conv_out.bn1.num_batches_tracked", "head.up2.conv_up.weight"]
    for key in to_delete:
        del checkpoint['state_dict'][key]
    #print(checkpoint['state_dict'].keys())
    #print("="*30)
    #print(checkpoint['state_dict'].keys())
    model.load_state_dict(checkpoint['state_dict'])
    return model

def colorize_mask(mask):

    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
               0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def data_transform(img, im_size, mean, std):
    from torchvision.transforms import functional as FT
    img = img.resize(im_size, Image.BILINEAR)
    tensor = FT.to_tensor(img)  # convert to tensor (values between 0 and 1)
    tensor = FT.normalize(tensor, mean, std)  # normalize the tensor
    return tensor


import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


ious_faulty = defaultdict(list)
ious_free = defaultdict(list)
ious_corrected = defaultdict(list)

def compute_iou(pred, target, num_classes=19):
    ious = {}
    mask = ~np.isnan(target)
    mask &= (target != 0)
    pred = pred.flatten()
    target = target.flatten()
    mask = mask.flatten() 
    pred = pred[mask]
    target = target[mask]
    for cls in np.unique(target):
        if cls == 0: continue
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union > 0:
            ious[cls] = intersection / union
    return ious

def visulization(args, model, model_last, device):
    # output folder
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # data transforms
    MEAN = (.485, .456, .406)
    STD = (.229, .224, .225)
   
    
    with torch.no_grad():
        image_list = glob.glob(os.path.join(args.test_folder, "*"))
        header = 'Demo'
        for i, imgName in tqdm(enumerate(image_list)):
            img = Image.open(imgName).convert('RGB')
            w, h = img.size
            tw, th = int(w*args.scale), int(h*args.scale) 
            scale_image = data_transform(img,(tw, th) , MEAN, STD)
            scale_image = scale_image.unsqueeze(0).to(device)
            residual, c1, c2 = model(scale_image)
            # quantizer2 = torch_quantizer(args.quant_mode, model_last, (c1, c2), output_dir=args.quant_dir,device=device)
            # quant_model2 = quantizer2.quant_model
            path = "data/split_bn1/fpn_mob_raw_out_0_103097_103098_400_1.csv"
            
            output = model_last(residual, c1, get_corrected_data(path))
            if isinstance(output, (tuple, list)):
                output = output[0]
            output = F.interpolate(output, size=(1024, 2048), mode='bilinear', align_corners=True)
            output = output.cpu().data[0].numpy().transpose(1,2,0)
            seg_pred = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            name = imgName.split('/')[-1]
            img_extn = imgName.split('.')[-1]
            color_mask = colorize_mask(seg_pred)
            color_mask.save(os.path.join(args.save_dir, path[15:]+'_CORRECTED_color.png'))

            output = model_last(residual, c1, get_faulty_data(path))
            if isinstance(output, (tuple, list)):
                output = output[0]
            output = F.interpolate(output, size=(1024, 2048), mode='bilinear', align_corners=True)
            output = output.cpu().data[0].numpy().transpose(1,2,0)
            seg_pred = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            name = imgName.split('/')[-1]
            img_extn = imgName.split('.')[-1]
            color_mask = colorize_mask(seg_pred)
            color_mask.save(os.path.join(args.save_dir, path[15:]+'_FAULTY_color.png'))

            output = model_last(residual, c1, get_free_data(path))
            if isinstance(output, (tuple, list)):
                output = output[0]
            output = F.interpolate(output, size=(1024, 2048), mode='bilinear', align_corners=True)
            output = output.cpu().data[0].numpy().transpose(1,2,0)
            seg_pred = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            name = imgName.split('/')[-1]
            img_extn = imgName.split('.')[-1]
            color_mask = colorize_mask(seg_pred)
            color_mask.save(os.path.join(args.save_dir, path[15:]+'_FREE_color.png'))

            

                
    


def eval_miou(data, model, device):
    from code.utils import miou_utils as utils
    #confmat = utils.ConfusionMatrix(args.num_classes)
    tbar = tqdm(data, desc='\r')
    with torch.no_grad():
        for i, (image, target) in enumerate(tbar): 
            image, target = image.to(device), target.to(device)
            output = model(image)
            
            #print(output.shape)
            '''
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.size()[2:] != target.size()[1:]:
                output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
            '''
            #confmat.update(target.flatten(), output.argmax(1).flatten())

        #confmat.reduce_from_all_processes()
    print('Evaluation Metric: ')
    #print(confmat)
    print("done")

def main(args):
    if args.dump_xmodel:
        args.device='cpu'
        args.batch_size=1

    if args.device=='cpu':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    # model
    model = build_model(args, device)
    model_last = build_model2(args, device)
    model.eval()


    #pytorch_total_params = sum(p.numel() for p in model.parameters())
    #print("Total number of parameters",pytorch_total_params)
    #model.to(device)
    #summary(model2, input_size=(3,512,1024))
    #print(model)

    H, W = args.crop_size, 2*args.crop_size
    input = torch.randn([1, 3, H, W])

    if args.quant_mode == 'float':
        quant_model = model
    else:
        ## new api
        ####################################################################################
        quantizer = torch_quantizer(args.quant_mode, model, (input), output_dir=args.quant_dir,device=device)
        quant_model = quantizer.quant_model
        #quantizer2 = torch_quantizer(args.quant_mode, model2, (input), output_dir="quantized_split",device=device)
        #quant_model2 = quantizer2.quant_model

    criterion = Criterion(ignore_index=255, weight=None, use_weight=False, reduce=True)
    loss_fn = criterion.to(device)
    
    # checkpoint_quant = torch.load(args.quant_dir+"/param.pth", map_location=device)
    # to_delete=["module_112.weight", "module_112.bias", "module_114.weight", "module_114.bias", "module_115.weight", "module_115.bias", "module_119.weight", "module_119.bias"]
    # for module in to_delete:
    #     del checkpoint_quant[module]
    # torch.save(checkpoint_quant, "quantized/param.pth")

    #checkpoint_bias_quant = torch.load(args.quant_dir+"/bias_corr.pth", map_location=device)
 
    #print(checkpoint_bias_quant) 


    if args.fast_finetune == True:
        print("This generates the FPN.py")
        ft_data = build_data(args, subset_len=None, sample_method=None)
        if args.quant_mode == 'calib':
            quantizer.fast_finetune(eval_miou, (ft_data, quant_model, device))
        elif args.quant_mode == 'test':
            print("testing")
            quantizer.load_ft_param()
    
    '''
    if args.quant_mode == 'calib' and args.finetune == True:
        ft_loader = build_data(args)
        quantizer.finetune(eval_miou, (quant_model, ft_loader, loss_fn))
    '''
    if args.eval:
        print('===> Evaluation mIoU: ')
        test_data = build_data(args)
        eval_miou(test_data, quant_model, device)    
    else:
        print('===> Visualization: ')
        visulization(args, quant_model, model_last, device)
        # Remove class 19 from the set of classes
        classes = sorted(set(ious_faulty.keys()) | set(ious_free.keys()) | set(ious_corrected.keys()))
        classes = [c for c in classes if c != 19]

        # Build IoU lists for the filtered classes
        min_faulty = [min(ious_faulty[c]) if c in ious_faulty else 0 for c in classes]
        min_free = [min(ious_free[c]) if c in ious_free else 0 for c in classes]
        min_corrected = [min(ious_corrected[c]) if c in ious_corrected else 0 for c in classes]

        # Plotting
        plt.rcParams['font.size'] = 20  # Set desired font size globally
        plt.rcParams['font.weight'] = 'bold'
        plt.figure(figsize=(12, 6))
        plt.plot(classes, min_faulty, label='Faulty', color="red")
        plt.plot(classes, min_free, label='Free', color="blue")
        plt.plot(classes, min_corrected, label='Corrected', color="green")
        #plt.xlabel("Class")
        plt.ylabel("Minimum IoU", weight="bold")
        plt.title("Minimum IoU per Class", fontweight="bold")
        #plt.legend(loc="lower left")  # or loc=3
        plt.grid(False)

        # Set custom xtick labels (corresponding to classes 0–13 only)
        plt.xticks(range(14), [
            "road", "sidewalk", "building", "wall", "fence", "pole", 
            "traffic light", "traffic sign", "vegetation", "terrain", 
            "sky", "person", "rider", "car"
        ], rotation=45)


        plt.tight_layout()
        plt.savefig("min_iou_per_class.png", dpi=300)


    # handle quantization result
    if args.quant_mode == 'calib':
        quantizer.export_quant_config()
    if args.quant_mode == 'test' and args.dump_xmodel:
        #deploy_check= True if args.dump_golden_data else False
        print("ok")
        dump_xmodel(args.quant_dir, deploy_check=True)

if __name__ == "__main__":
    args = Configs().parse()
    torch.manual_seed(args.seed)
    main(args)
