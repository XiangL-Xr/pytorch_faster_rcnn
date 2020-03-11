# !/usr/bin/python
# coding : utf-8
# Author : lixiang
# Func   : increg pruning methods
import os, time
import math
import torch as t
import numpy as np
import torch.nn as nn

# Integrate the idea of IncReg to Pruning --------------------
class SparseRegularization(object):

    def __init__(self, args):
        self.init_hyparams()
        self.init_regular()
        
        self.state = args.state
        self.lr = args.lr
        self.flag = False
        self.IF_prune_finished = False
    
    def init_hyparams(self):
        self.kk = 0.25
        self.AA = 0.00025
        self.prune_iter = -1
        self.prune_interval = 1
        self.print_freq = 20
        self.target_reg = 1.0
    
    def init_regular(self):
        self.skip_idx = []
        self.compress_rate = {}
        self.history_score = {}
        self.Reg = {}
        self.masks = {}
        self.pruned_rate = {}
        self.num_pruned_col = {}
        self.IF_col_alive = {}
        self.IF_col_pruned = {}
        self.IF_layer_finished = {}

    def init_rate(self, args, model):
        conv_cnt = -1
        self.init_skiplayer(args)
        #for m in model.modules():
        for p in model.parameters():
            if len(p.data.size()) == 4:
            #if isinstance(m, nn.Conv2d):
                #print("-----p.size:", p.data.size())
                conv_cnt += 1; index = str(conv_cnt)
                if conv_cnt in [0, 1, 2]:
                    continue
                self.IF_layer_finished[index] = 0
                if not args.skip:
                    self.compress_rate[index] = args.rate
                else:
                    self.compress_rate[index] = 0 if (conv_cnt in self.skip_idx) else args.rate

                print("Conv_" + index + " initialize compress ratio:", self.compress_rate[index])
    
    def init_skiplayer(self, args):
        if args.skip and (args.net == "vgg16"):
            self.skip_idx = [3, 15] if (args.rate > 0.5) else [3, 13, 14, 15]
        elif args.skip and (args.net == "resnet50"):
            self.skip_idx = [0]
        print(args.net + " skip layer initialization: ", self.skip_idx)

    def init_register(self, index, p, num_col):
        # initialization/registrztion
        if index not in self.Reg:
            self.Reg[index] = [0] * num_col
            self.masks[index] = [1] * num_col
            self.IF_col_alive[index] = [1] * num_col
            self.num_pruned_col[index] = 0
            self.pruned_rate[index] = 0
            self.history_score[index] = [0] * num_col
            self.init_mask(index, p)
            if self.compress_rate[index] == 0:
                self.IF_layer_finished[index] = 1
        
        num_pruned_col_ = self.num_pruned_col[index]
        num_col_ = num_col - num_pruned_col_
        num_col_to_prune_ = math.ceil(num_col * self.compress_rate[index]) - num_pruned_col_
        return num_col_, num_col_to_prune_

    def init_length(self, p):
        N = p.data.shape[0]
        C = p.data.shape[1]
        H = p.data.shape[2]
        W = p.data.shape[3]
        return N, C, H, W

    def update_increg(self, model):
        conv_cnt = -1
        #for m in model.modules():
        for p in model.parameters():
            if len(p.data.size()) == 4:
                conv_cnt += 1; index = str(conv_cnt)
                if conv_cnt in [0, 1, 2]:
                    continue
                N, C, H, W = self.init_length(p)
                num_col = C * H * W
                W_data = np.array(p.data.cpu()).reshape(N, -1)
                num_col_, num_col_to_prune_ = self.init_register(index, p, num_col)
                #print("----p_data:", p.data)
                #print("----p_grad:", p.grad)
                # if all conv layers pruning finished
                if all(self.IF_layer_finished.values()):
                    self.IF_prune_finished = True
                    continue
                elif self.IF_layer_finished[index] == 1:
                    p.data.mul_(self.masks[index])
                    if p.requires_grad:
                        p.grad.mul_(self.masks[index])
                    continue
                    
                """ ## Start pruning """
                if num_col_to_prune_ > 0:
                    # step 01: get importance ranking
                    col_hist_rank = self.get_rank_score(index, W_data, num_col)
                    # step 02: sparse regularization
                    self.do_regular(index, p, num_col_, num_col_to_prune_, num_col, col_hist_rank)
                    # step 03: get mask matrix
                    self.get_mask(index, p)

                """ ## Mask out the gradient and weights """
                if p.requires_grad:
                    p.grad.mul_(self.masks[index])
                p.data.mul_(self.masks[index])

    
    def get_rank_score(self, index, W_data, num_col):
        # execute once every one step
        if self.prune_iter % self.prune_interval == 0:
            """ ### Sort 01: sort by L1-norm """
            col_score = {}
            col_score_first = []
            col_score_second = []
            for j in range(num_col):
                col_score_first.append(j)
                col_score_second.append(np.sum(np.fabs(W_data[:, j])))
            col_score = dict(zip(col_score_first, col_score_second))
            col_score_rank = sorted(col_score.items(), key = lambda k: k[1])

            # Make new criteria, i.e. history_rank, by rank
            # No.n iter, n starts from 1
            n = self.prune_iter + 1                     
            for rk in range(num_col):
                col_of_rank_rk = col_score_rank[rk][0]
                self.history_score[index][col_of_rank_rk] = ((n - 1) * self.history_score[index][col_of_rank_rk] + rk) / n

            """ ### Sort 02: sort by history rank """
            # the history_rank of each column, history_rank is like the new score
            col_hrank = {}
            col_hrank_first = []
            col_hrank_second = []
            for j in range(num_col):
                col_hrank_first.append(j)
                col_hrank_second.append(self.history_score[index][j])
            col_hrank = dict(zip(col_score_first, col_score_second))
            col_hist_rank = sorted(col_hrank.items(), key = lambda k: k[1])

        return col_hist_rank

    def do_regular(self, index, p, num_col_, num_col_to_prune_, num_col, col_hist_rank):
        # Note the real rank is i + num_pruned_col_
        num_pruned_col_ = num_col - num_col_
        for i in range(num_col_):
            col_of_rank_i = col_hist_rank[i + num_pruned_col_][0]
            Delta = self.punish_func(i, num_col_, num_col_to_prune_)
            self.Reg[index][col_of_rank_i] = max(self.Reg[index][col_of_rank_i] + Delta, 0)
        
            if self.Reg[index][col_of_rank_i] >= self.target_reg:
                self.IF_col_alive[index][col_of_rank_i] = 0
                self.num_pruned_col[index] += 1
                self.pruned_rate[index] = self.num_pruned_col[index] / num_col  
                if self.pruned_rate[index] >= self.compress_rate[index]:
                    self.IF_layer_finished[index] = 1        
        
        """ ### Apply reg to conv weights """
        _, C, H, W = self.init_length(p)
        Reg_tmp = np.array(self.Reg[index]).reshape(-1, C, H, W)
        Reg_new = t.from_numpy(Reg_tmp).float().cuda()
        # use L2 regularization
        if p.requires_grad:
            p.grad.add_(Reg_new * p.data)

    def init_mask(self, index, p):
        # initialization mask to 4-dim tensor
        _, C, H, W = self.init_length(p)
        tmp = np.array(self.masks[index]).reshape(-1, C, H, W)
        self.masks[index] = t.from_numpy(tmp).float().cuda()

    def check_mask(self, model):
        conv_cnt = -1
        #for m in model.modules():
        for p in model.parameters():
            if len(p.data.size()) == 4:
                conv_cnt += 1; index = str(conv_cnt)
                _, C, H, W = self.init_length(m)
                if index not in self.masks:
                    self.masks[index] = [1] * (C * H * W)
                mask = (m.weight.data != 0)
                self.masks[index] = mask.float().cuda()
        
    def get_mask(self, index, p):
        # generate mask matrix
        _, C, H, W = self.init_length(p)
        tmp = np.array(self.IF_col_alive[index]).reshape(-1, C, H, W)
        self.masks[index] = t.from_numpy(tmp).float().cuda()

    def do_mask(self, model):
        conv_cnt = -1
        for p in model.parameters():
            if len(p.data.size()) == 4:
                conv_cnt += 1; index = str(conv_cnt)
                if conv_cnt in [0, 1, 2]:
                    continue
                if p.requires_grad:
                    p.grad.mul_(self.masks[index])
                p.data.mul_(self.masks[index])
                #print("self.masks:", self.masks[index])
    
    def reg_pruning(self, model, epoch):
        if self.state == "prune":
            start_time = time.time()
            self.prune_iter += 1
            self.update_increg(model)
            batch_time = time.time() - start_time
            self.info_print(epoch, batch_time)
            
            if self.IF_prune_finished:
                self.flag = True
                self.info_print(epoch, batch_time)
                self.state = "retrain"
                print("=> pruning stage finished, start retrain...")
        
        elif self.state == "retrain":
            self.do_mask(model)

    def info_print(self, epoch, batch_time):
        if (self.prune_iter % self.print_freq) == 0 or self.flag:
            for key, value in self.pruned_rate.items():
                print('--[Train->prune]-- epoch:[{0}], iter:[{1}], conv-{key:}\t'
                      'rate: [{value:.3f}/{PR:.3f}], lr: #{lr:}#\t'
                      'time: {batch_time:.3f}\t'
                      .format(epoch,
                              self.prune_iter,
                              key = key,
                              value = value,
                              PR = self.compress_rate[key],
                              lr = self.lr,
                              batch_time = batch_time))

    # Punish Function 1 ----------------
    def punish_scheme1(self, r, num_col_to_prune_):
        alpha = math.log(2/self.kk) / num_col_to_prune_
        N = -math.log(self.kk) / alpha
        if r < N:
            return self.AA * math.exp(-alpha*r)
        else:
            return (2*self.kk*self.AA) - (self.AA*math.exp(-alpha*(2*N-r)))
    
    # Punish Function 2 ----------------
    def punish_func(self, r, num_g, thre_rank):
        if r <= thre_rank:
            return self.AA - (self.AA/thre_rank*r)
        else:
            return -self.AA / (num_g-1-thre_rank) * (r-thre_rank)
