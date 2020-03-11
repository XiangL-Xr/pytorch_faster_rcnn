# !/usr/bin/python
# coding : utf-8
# Author : lixiang
import os, sys
sys.path.append('../')

import time
import torch as t
import numpy as np
import torch.nn as nn

def Compute_Rowratio(model):    
    conv_cnt = -1
    conv_row_rate = []
    for p in model.parameters():
        if len(p.data.size()) == 4:
            conv_cnt += 1
            if conv_cnt in [0, 1, 2]:
                continue
            interval = p.data.shape[2] * p.data.shape[3]
            row_nums = p.data.shape[0]
            param_np = np.array(p.data.cpu()).reshape(p.data.shape[0], -1)
            sum_col_params = 0
            masks = []
            for i in range(param_np.shape[1]):
                sum_col_params += np.sum(np.abs(param_np[:, i]))
                #print("sum_col_params", sum_col_params)
                if (i+1) % interval == 0:
                    mask = (sum_col_params != 0)
                    masks.append(float(mask))
                    sum_col_params = 0
            
            np_mask = np.array(masks)
            row_rate = 1 - np.sum(np_mask) / len(masks)
            conv_row_rate.append(row_rate)
    
    conv_row_rate.append(0)
    return conv_row_rate

def Prune_rate_compute(model, verbose = True):
    """ Print out prune rate for each layer and the whole network """
    row_prune_rate = Compute_Rowratio(model)
    total_rows = 0
    total_cols = 0
    zero_cols = 0
    conv_cnt = -1
    row_cnt = 0
    for p in model.parameters():        
        layer_zero_col_count = 0
        if len(p.data.size()) == 4:
            conv_cnt += 1
            if conv_cnt in [0, 1, 2]:
                continue
            layer_row_nums = p.data.shape[0]
            layer_col_nums = p.data.shape[1] * p.data.shape[2] * p.data.shape[3]
            param_np = np.array(p.data.cpu()).reshape(p.data.shape[0], -1)
            for idx in range(layer_col_nums):
                if np.sum(np.abs(param_np[:, idx])) == 0:
                    layer_zero_col_count += 1
            
            total_rows += layer_row_nums
            total_cols += layer_col_nums
            zero_cols += layer_zero_col_count
        
            layer_col_prune_rate = float(layer_zero_col_count) / layer_col_nums
        
            if verbose:
                row_cnt += 1
                print("-----------------------------------------------------------------")
                print("Layer {} | {} layer | {:.2f}% rows pruned | {:.2f}% cols pruned"
                      .format(conv_cnt, 'Conv', 100.*row_prune_rate[row_cnt], 100.*layer_col_prune_rate))

    col_prune_rate = float(zero_cols) / total_cols
    
    if verbose:
        print("==================== Params Compress Rate =======================")
        #print("Final row pruning rate: {:.2f}".format(row_prune_rate))
        print("Final col pruning rate: {:.2f}".format(col_prune_rate))
        #print("Final params pruning rate: {:.2f}".format(prune_rate))
        print("=================================================================")    
    #return row_prune_rate, col_prune_rate

# use to compute pruned model gflops(->compute_flops.py)
def layer_prune_rate(m):
    layer_zero_row_count = 0
    layer_zero_col_count = 0
    N, C, H, W = init_length(m)
    layer_row_nums = N
    layer_col_nums = C * H * W
    param_np = np.array(m.weight.data.cpu()).reshape(N, -1)
    for idx in range(layer_row_nums):
        if np.sum(np.abs(param_np[idx, :])) == 0:
            layer_zero_row_count += 1
    for idx in range(layer_col_nums):
        if np.sum(np.abs(param_np[:, idx])) == 0:
            layer_zero_col_count += 1

    layer_row_prune_rate = float(layer_zero_row_count) / layer_row_nums
    layer_col_prune_rate = float(layer_zero_col_count) / layer_col_nums
    return layer_row_prune_rate, layer_col_prune_rate


""" ---------------------- model flops calculation -----------------------"""
def check_flops(args, model):
    get_flops(model)
    input = t.zeros(1, 3, 32, 32)
    if args.use_gpu:
        input = input.cuda()
    out = model(input)
    conv_flops_ = get_conv_flops(model)
    total_flops = sum(list_conv) + sum(list_linear)
    last_flops = conv_flops_ + sum(list_linear)
    return total_flops, last_flops

def get_flops(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(conv_hook)
        if isinstance(m, nn.Linear):
            m.register_forward_hook(linear_hook)

def get_conv_flops(model):
    conv_idx = -1
    conv_flops = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            conv_idx += 1
            if conv_idx in [0, 1, 2]:
                continue
            row_rate, col_rate = layer_prune_rate(m)
            N, C, H, W = init_length(m)
            num_row_ = N * (1 - row_rate)
            num_col_ = C * H * W * (1 - col_rate)
            params_ = num_row_ * num_col_
            gflops_ = params_ * list_conv_fmap[conv_idx] ** 2
            conv_flops += gflops_
    return conv_flops

def init_length(m):
    N = m.weight.data.shape[0]
    C = m.weight.data.shape[1]
    H = m.weight.data.shape[2]
    W = m.weight.data.shape[3]
    return N, C, H, W

# hook list initialization
list_conv = []
list_linear = []
list_conv_fmap = []

# hook calculation
def conv_hook(self, input, output):
    batch_size, input_channels, input_height, input_width = input[0].size()
    output_channels, output_height, output_width = output[0].size()
    kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
    bias_ops = 1 if self.bias is not None else 0
    params = output_channels * (kernel_ops + bias_ops)
    flops = batch_size * params * output_height * output_width
    list_conv.append(flops)
    list_conv_fmap.append(output_height)

def linear_hook(self, input, output):
    batch_size = input[0].size(0) if input[0].dim() == 2 else 1
    weight_ops = self.weight.nelement()
    bias_ops = self.bias.nelement()
    flops = batch_size * (weight_ops + bias_ops)
    list_linear.append(flops)
