# !/usr/bin/python
# coding : utf-8
# Author : lixiang
import os 
os.environ['CUDA_VISIBLE_DEVICES']='5'

import _init_paths
import torch as t
import torch.nn as nn
import numpy as np

from Pruner.prune_engine import *
from model.faster_rcnn.vgg16 import vgg16
from roi_data_layer.roidb import combined_roidb
from model.utils.config import cfg

import argparse
parser = argparse.ArgumentParser(description='Pytorch Example')

parser.add_argument('--net', default="vgg16", type=str,
                    help='model selection, choices: vgg16, resnet50')
parser.add_argument('--dataset', default="pascal_voc", type=str,
                    help='dataset selection')
parser.add_argument('--use_gpu', default=True, type=bool, metavar='N',
                    help='use gpu or not, (default:True)')
parser.add_argument('--weights', default='weights/model.pth', type=str,
                    help='path to pickled weights')
parser.add_argument('--weight_group', default="Col", type=str,
                    help='pre prune basic units, default=filter')
parser.add_argument('--IF_update_row_col', default=True, type=bool,
                    help='IF update row sparse ratio or col sparse ratio')
parser.add_argument('--IF_save_update_model', default=False, type=bool,
                    help='IF save model of updated row or col sparse ratio')

parser.add_argument('--cag', dest='class_agnostic',
                    help='whether to perform class_agnostic bbox regression',
                    action='store_true')

args = parser.parse_args()

if args.dataset == "pascal_voc":
    args.imdb_name = "voc_2007_trainval"
    args.imdbval_name = "voc_2007_test"
    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)

if args.net == 'vgg16':
    model = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
model.create_architecture()

if args.weights:
    if os.path.isfile(args.weights):
        print("load checkpoint %s" % (args.weights))
        checkpoint = t.load(args.weights)
        model.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
    
        print('load model successfully!')
    else:
        print("=> no checkpoint found at '{}'".format(args.weights))

if args.use_gpu:
    model.cuda()

#total_flops, last_flops = check_flops(args, model)
Prune_rate_compute(model, verbose = True)
#print("==================== Flops Compress Rate ========================")
#print('Total number of flops: {:.2f}G'.format(total_flops / 1e9))
#print('Last number of flops: {:.2f}G'.format(last_flops / 1e9))
#print("Final model speedup rate: {:.2f}".format(total_flops / last_flops))
#print("=================================================================")
