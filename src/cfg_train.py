#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ************************************
# @Time     : 2019/7/3 22:34
# @Author   : Anonymous
# @File     : cfg_train.py
# ************************************
import os
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import auc, roc_curve

from cfg_config import cfg_args
from data import CFGDataset
from model.DenseGraphMatching import HierarchicalGraphMatchNetwork
from utils import create_dir_if_not_exists, write_log_file
from utils import generate_epoch_pair


class CFGTrainer(object):
    def __init__(self, node_init_dims, data_dir, device, log_file, best_model_file, args):
        super(CFGTrainer, self).__init__()
        
        # training parameters
        self.max_epoch = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.device = device
        
        self.log_file = log_file
        self.best_model_path = best_model_file
        
        self.model = HierarchicalGraphMatchNetwork(node_init_dims=node_init_dims, arguments=args, device=device).to(
            device)
        write_log_file(self.log_file, str(self.model))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        cfg = CFGDataset(data_dir=data_dir, batch_size=self.batch_size)
        
        self.graph_train = cfg.graph_train
        self.classes_train = cfg.classes_train
        self.epoch_data_valid = cfg.valid_epoch
        self.epoch_data_test = cfg.test_epoch
        
        init_val_auc = self.eval_auc_epoch(model=self.model,
                                           eval_epoch_data=self.epoch_data_valid)  # evaluate the auc of init model for validation dataset
        write_log_file(self.log_file, "initial validation AUC = {0} @ {1}".format(init_val_auc, datetime.now()))
    
    def fit(self):
        best_val_auc = None
        for i in range(1, self.max_epoch + 1):
            # train
            loss_avg = self.train_one_epoch(model=self.model, optimizer=self.optimizer, graphs=self.graph_train,
                                            classes=self.classes_train, batch_size=self.batch_size,
                                            device=self.device, load_data=None)
            write_log_file(self.log_file,
                           "EPOCH {0}/{1}:\tMSE loss = {2} @ {3}".format(i, self.max_epoch, loss_avg, datetime.now()))
            # validation
            valid_auc = self.eval_auc_epoch(model=self.model, eval_epoch_data=self.epoch_data_valid)
            write_log_file(self.log_file, "validation AUC = {0} @ {1}".format(valid_auc, datetime.now()))
            if best_val_auc is None or best_val_auc < valid_auc:
                write_log_file(self.log_file,
                               '\tvalidation AUC increased ({} ---> {}), and saving the model ... '.format(best_val_auc,
                                                                                                           valid_auc))
                best_val_auc = valid_auc
                torch.save(self.model.state_dict(), self.best_model_path)
            
            write_log_file(self.log_file, '\tbest validation auc = {} '.format(best_val_auc))
        return best_val_auc
    
    def testing(self):
        # load the last checkpoint with the best model
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.model.eval()
        # double check the save checkpoint model for validation
        double_val_auc = self.eval_auc_epoch(model=self.model, eval_epoch_data=self.epoch_data_valid)
        # evaluating on the testing dataset
        final_test_auc = self.eval_auc_epoch(model=self.model, eval_epoch_data=self.epoch_data_test)
        
        write_log_file(self.log_file,
                       "\n\nDouble check for the saved best checkpoint model for validation {} ".format(double_val_auc))
        write_log_file(self.log_file, "Finally, testing auc = {} @ {}".format(final_test_auc, datetime.now()))
        return final_test_auc
    
    @staticmethod
    def train_one_epoch(model, optimizer, graphs, classes, batch_size, device, load_data=None):
        model.train()
        if load_data is None:
            epoch_data = generate_epoch_pair(graphs, classes, batch_size)
        else:
            epoch_data = load_data
        
        perm = np.random.permutation(len(epoch_data))  # Random shuffle
        
        cum_loss = 0.0
        num = 0
        for index in perm:
            cur_data = epoch_data[index]
            x1, x2, adj1, adj2, y = cur_data
            batch_output = model(batch_x_p=x1, batch_x_h=x2, batch_adj_p=adj1, batch_adj_h=adj2)
            y = torch.FloatTensor(y).to(device)
            mse_loss = torch.nn.functional.mse_loss(batch_output, y)
            
            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()
            
            cum_loss += mse_loss
            if num % int(len(perm) / 10) == 0:
                print('\tTraining: {}/{}: index = {} loss = {}'.format(num, len(epoch_data), index, mse_loss))
            num = num + 1
        return cum_loss / len(perm)
    
    @staticmethod
    def eval_auc_epoch(model, eval_epoch_data):
        model.eval()
        with torch.no_grad():
            tot_diff = []
            tot_truth = []
            for cur_data in eval_epoch_data:
                x1, x2, adj1, adj2, y = cur_data
                batch_output = model(batch_x_p=x1, batch_x_h=x2, batch_adj_p=adj1, batch_adj_h=adj2)
                
                tot_diff += list(batch_output.data.cpu().numpy())
                tot_truth += list(y > 0)
        
        diff = np.array(tot_diff) * -1
        truth = np.array(tot_truth)
        
        fpr, tpr, _ = roc_curve(truth, (1 - diff) / 2)
        model_auc = auc(fpr, tpr)
        return model_auc


if __name__ == '__main__':
    d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg_args.gpu_index)
    
    main_data_dir = cfg_args.data_dir
    graph_name = cfg_args.dataset
    graph_min = cfg_args.graph_size_min
    graph_max = cfg_args.graph_size_max
    graph_init_dim = cfg_args.graph_init_dim
    
    # <-><-><-> only for log, delete below if open source
    title = '{}_Min{}_Max{}_InitDims{}_Task_{}_Filter_{}_Match_{}_P_{}_Agg_{}_Hidden_{}_Epoch_{}_Batch_{}_lr_{}_Dropout_{}_Global_{}_with_agg_{}' \
        .format(graph_name,
                graph_min,
                graph_max,
                graph_init_dim,
                cfg_args.task,
                cfg_args.filters,
                cfg_args.match,
                cfg_args.perspectives,
                cfg_args.match_agg,
                cfg_args.hidden_size,
                cfg_args.epochs,
                cfg_args.batch_size,
                cfg_args.lr,
                cfg_args.dropout,
                int(cfg_args.global_flag),
                cfg_args.global_agg)
    main_log_dir = cfg_args.log_path + '{}_Min{}_Max{}_InitDims{}_Task_{}/'.format(graph_name, graph_min, graph_max, graph_init_dim, cfg_args.task)
    create_log_str = create_dir_if_not_exists(main_log_dir)
    best_model_dir = main_log_dir + 'BestModels_Repeat_{}/'.format(cfg_args.repeat_run)
    create_BestModel_dir = create_dir_if_not_exists(best_model_dir)
    LOG_FILE = main_log_dir + 'repeat_{}_'.format(cfg_args.repeat_run) + title + '.txt'
    BestModel_FILE = best_model_dir + title + '.BestModel'
    CSV_FILE = main_log_dir + title + '.csv'
    
    write_log_file(LOG_FILE, create_log_str)
    write_log_file(LOG_FILE, create_BestModel_dir)
    write_log_file(LOG_FILE, str(cfg_args))
    write_log_file(LOG_FILE, title)
    # <-><-><-> only for log, delete above if open source
    
    sub_data_dir = '{}_{}ACFG_min{}_max{}'.format(graph_name, graph_init_dim, graph_min, graph_max)
    cfg_data_dir = os.path.join(main_data_dir, sub_data_dir) if 'ffmpeg' in sub_data_dir else os.path.join(main_data_dir, sub_data_dir, 'acfgSSL_6')
    assert os.path.exists(cfg_data_dir), "the path of {} is not exist!".format(cfg_data_dir)
    
    cfg_trainer = CFGTrainer(node_init_dims=graph_init_dim, data_dir=cfg_data_dir, device=d, log_file=LOG_FILE, best_model_file=BestModel_FILE, args=cfg_args)
    ret_best_val_auc = cfg_trainer.fit()
    ret_final_test_auc = cfg_trainer.testing()
    
    # <-><-><-> only for log, delete below if open source
    log_str = "Index,Min,Max,InitDims,Perspectives,HiddenSize,Epoch,Batch,LR,Dropout,Sep,BestValAUC,FinalTestAUC\n"
    log_data = np.asarray([[cfg_args.repeat_run, graph_min, graph_max, graph_init_dim, cfg_args.perspectives, cfg_args.hidden_size, cfg_args.epochs, cfg_args.batch_size,
                            cfg_args.lr, cfg_args.dropout, 99999, ret_best_val_auc, ret_final_test_auc]])
    if os.path.isfile(CSV_FILE):
        with open(CSV_FILE, 'a+') as csv_file:
            np.savetxt(csv_file, log_data, delimiter=', ', newline='\n')
    else:
        with open(CSV_FILE, 'w+') as csv_file:
            csv_file.write('{}\n'.format(cfg_data_dir))
            csv_file.write(log_str)
            np.savetxt(csv_file, log_data, delimiter=', ', newline='\n')
    # <-><-><-> only for log, delete below if open source
