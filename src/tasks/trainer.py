#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:53:55 2019

@author: weetee
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from src.triplet import TrippletModel
from testing import eval_sebert
from .preprocessing_funcs import load_dataloaders
from .train_funcs import load_state, load_results, evaluate_, evaluate_results, l2
from ..misc import save_as_pickle, load_pickle
import matplotlib.pyplot as plt
import time
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def eval_max(l, size):
    for i in l:
        if len(i) > size:
            return True
    return False

def train_and_fit(args):
    
    if args.fp16:    
        from apex import amp
    else:
        amp = None
    
    cuda = torch.cuda.is_available()
    
    train_loader, test_loader, train_len, test_len = load_dataloaders(args)
    # eval_sebert(test_loader)
    logger.info("Loaded %d Training samples." % train_len)
    
    if args.model_no == 0:
        from ..model.BERT.modeling_bert import BertModel as Model
        model = args.model_size #'bert-base-uncased'
        lower_case = True
        model_name = 'BERT'
        net = Model.from_pretrained(model, force_download=False, \
                                model_size=args.model_size,
                                task='classification' if args.task != 'fewrel' else 'fewrel',\
                                n_classes_=args.num_classes)
    elif args.model_no == 1:
        from ..model.ALBERT.modeling_albert import AlbertModel as Model
        model = args.model_size #'albert-base-v2'
        lower_case = True
        model_name = 'ALBERT'
        net = Model.from_pretrained(model, force_download=False, \
                                model_size=args.model_size,
                                task='classification' if args.task != 'fewrel' else 'fewrel',\
                                n_classes_=args.num_classes)
    elif args.model_no == 2: # BioBert
        from ..model.BERT.modeling_bert import BertModel, BertConfig
        model = 'bert-base-uncased'
        lower_case = False
        model_name = 'BioBERT'
        config = BertConfig.from_pretrained('./additional_models/biobert_v1.1_pubmed/bert_config.json')
        net = BertModel.from_pretrained(pretrained_model_name_or_path='./additional_models/biobert_v1.1_pubmed/biobert_v1.1_pubmed.bin', 
                                          config=config,
                                          force_download=False, \
                                          model_size='bert-base-uncased',
                                          task='classification' if args.task != 'fewrel' else 'fewrel',\
                                          n_classes_=args.num_classes)
    elif args.model_no == 3:
        net = TrippletModel(args.model)

    if args.model_no == 3:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        tokenizer = load_pickle("%s_tokenizer.pkl" % model_name)
        net.resize_token_embeddings(len(tokenizer))
        e1_id = tokenizer.convert_tokens_to_ids('[E1]')
        e2_id = tokenizer.convert_tokens_to_ids('[E2]')
        assert e1_id != e2_id != 1
    
    if cuda:
        net.cuda()
        
    logger.info("FREEZING MOST HIDDEN LAYERS...")
    if args.model_no == 0:
        unfrozen_layers = ["classifier", "pooler", "encoder.layer.11", \
                           "classification_layer", "blanks_linear", "lm_linear", "cls"]
    elif args.model_no == 1:
        unfrozen_layers = ["classifier", "pooler", "classification_layer",\
                           "blanks_linear", "lm_linear", "cls",\
                           "albert_layer_groups.0.albert_layers.0.ffn"]
    elif args.model_no == 2:
        unfrozen_layers = ["classifier", "pooler", "encoder.layer.11", \
                           "classification_layer", "blanks_linear", "lm_linear", "cls"]
    elif args.model_no == 3:
        unfrozen_layers = ["classifier", "pooler", "encoder.layer.11", \
                           "classification_layer", "blanks_linear", "lm_linear", "cls", 'encoder.layer.10', 'encoder.layer.9"']
        
    for name, param in net.named_parameters():
        if not any([layer in name for layer in unfrozen_layers]):
            print("[FROZE]: %s" % name)
            param.requires_grad = False
        else:
            print("[FREE]: %s" % name)
            param.requires_grad = True
        # if param.requires_grad:
        #     print("[FREE]: %s" % name)
        # else:
        #     print("[FROZE]: %s" % name)
    if args.use_pretrained_blanks == 1:
        logger.info("Loading model pre-trained on blanks at ./data/test_checkpoint_%d.pth.tar..." % args.model_no)
        checkpoint_path = "./data/test_checkpoint_%d.pth.tar" % args.model_no
        checkpoint = torch.load(checkpoint_path)
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        net.load_state_dict(pretrained_dict, strict=False)
        del checkpoint, pretrained_dict, model_dict
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam([{"params":net.parameters(), "lr": args.lr}])
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,\
                                                                      24,26,30], gamma=0.8)
    
    start_epoch, best_pred, amp_checkpoint = load_state(net, optimizer, scheduler, args, load_best=False)  
    best_pred = 99
    if (args.fp16) and (amp is not None):
        logger.info("Using fp16...")
        net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
        if amp_checkpoint is not None:
            amp.load_state_dict(amp_checkpoint)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,\
                                                                          24,26,30], gamma=0.8)
    
    losses_per_epoch, accuracy_per_epoch, test_f1_per_epoch = load_results(args.model_no)

    logger.info("Starting training process...")
    update_size = min(len(train_loader)//20, args.val_step)
    # eval_sebert(test_loader, net)
    net.eval()
    # t_results = eval_sebert(test_loader, net)
    # print(f't_results: {t_results}')
    net.train()
    for epoch in tqdm(range(start_epoch, args.num_epochs)):
        start_time = time.time()
        net.train(); total_loss = 0.0; losses_per_batch = []; accuracy_per_batch = []
        for i, data in enumerate(train_loader, 0):
            # p0, p1, n, masks = data
            # p, n, masks = data
            p, n, n2, masks = data
            if eval_max(p, 512) or eval_max(n, 512):
                continue
            # if len(masks) != 3
            # t_results = eval_sebert(test_loader, net)
            # print(f't_results: {t_results}')
            if cuda:
                p = [x.cuda() for x in p]; n= [x.cuda() for x in n]
                n2 = [x.cuda() for x in n2]

                # loss = net.run_train_multi(p, n, masks)
                # print('p: {}, n:{}, mask:{}'.format([x.size() for x in p], [x.size() for x in n] , [x.size() for x in masks]))
            if any(x.size()[-1] > 512 for x in n) or any(x.size()[-1] > 512 for x in p): #skip long sentences
                continue
            if args.task == 'double_negative_similarity':
                loss = net.run_train_double(p, n, n2, masks)
            else:
                loss = net.run_train_multi_reduce(p, n, masks)

            loss.backward()
            grad_norm = clip_grad_norm_(net.parameters(), args.max_norm)
            
            if (i % args.gradient_acc_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            # total_acc += evaluate_(classification_logits, labels, \
            #                        ignore_idx=-1)[0]
            
            if (i % update_size) == (update_size - 1):
                losses_per_batch.append(args.gradient_acc_steps*total_loss/update_size)
                # accuracy_per_batch.append(total_acc/update_size)
                print('[Epoch: %d, %5d/ %d points] total loss, %.3f' %
                      (epoch + 1, (i + 1)*args.batch_size, train_len, losses_per_batch[-1]))
                total_loss = 0.0
            # if i+1 % args.val_step == 0:
                print('val step eval. Step {}'.format(i))
                # results = evaluate_results(net, test_loader, cuda)
                results = eval_sebert(test_loader, net)
                losses_per_epoch.append(sum(losses_per_batch) / len(losses_per_batch))
                print("Epoch finished, took %.2f seconds." % (time.time() - start_time))
                print("Losses at Epoch %d: %.7f" % (epoch + 1, losses_per_epoch[-1]))

                if results['loss'] < best_pred:
                    print("new best loss: {}!".format(results['loss']))
                    best_pred = results['loss']
                    torch.save({
                        'epoch': epoch + 1, \
                        'state_dict': net.state_dict(), \
                        'best_loss': losses_per_epoch[-1], \
                        'optimizer': optimizer.state_dict(), \
                        'scheduler': scheduler.state_dict(), \
                        'amp': amp.state_dict() if amp is not None else amp
                    }, os.path.join("./data/", "task_test_model_best_%d.pth.tar" % args.model_no))
        
        scheduler.step()
        # results = evaluate_results(net, test_loader, cuda)
        results = eval_sebert(test_loader, net)
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        print("Epoch finished, took %.2f seconds." % (time.time() - start_time))
        print("Losses at Epoch %d: %.7f" % (epoch + 1, losses_per_epoch[-1]))
        
        if results['loss'] < best_pred:
            print("new best loss: {}!".format(results['loss']))
            best_pred = results['loss']
            torch.save({
                    'epoch': epoch + 1,\
                    'state_dict': net.state_dict(),\
                    'best_loss': losses_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                    'amp': amp.state_dict() if amp is not None else amp
                }, os.path.join("./data/" , "task_test_model_best_%d.pth.tar" % args.model_no))
        
        if (epoch % 1) == 0:
            save_as_pickle("task_test_losses_per_epoch_%d.pkl" % args.model_no, losses_per_epoch)
            # save_as_pickle("task_train_accuracy_per_epoch_%d.pkl" % args.model_no, accuracy_per_epoch)
            save_as_pickle("task_test_f1_per_epoch_%d.pkl" % args.model_no, test_f1_per_epoch)
            torch.save({
                    'epoch': epoch + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': losses_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                    'amp': amp.state_dict() if amp is not None else amp
                }, os.path.join("./data/" , "task_test_checkpoint_%d.pth.tar" % args.model_no))
    
    logger.info("Finished Training!")
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    ax.scatter([e for e in range(len(losses_per_epoch))], losses_per_epoch)
    ax.tick_params(axis="both", length=2, width=1, labelsize=14)
    ax.set_xlabel("Epoch", fontsize=22)
    ax.set_ylabel("Training Loss per batch", fontsize=22)
    ax.set_title("Training Loss vs Epoch", fontsize=32)
    plt.savefig(os.path.join("./data/" ,"task_loss_vs_epoch_%d.png" % args.model_no))
    
    fig2 = plt.figure(figsize=(20,20))
    ax2 = fig2.add_subplot(111)
    ax2.scatter([e for e in range(len(accuracy_per_epoch))], accuracy_per_epoch)
    ax2.tick_params(axis="both", length=2, width=1, labelsize=14)
    ax2.set_xlabel("Epoch", fontsize=22)
    ax2.set_ylabel("Training Accuracy", fontsize=22)
    ax2.set_title("Training Accuracy vs Epoch", fontsize=32)
    plt.savefig(os.path.join("./data/" ,"task_train_accuracy_vs_epoch_%d.png" % args.model_no))
    
    fig3 = plt.figure(figsize=(20,20))
    ax3 = fig3.add_subplot(111)
    ax3.scatter([e for e in range(len(test_f1_per_epoch))], test_f1_per_epoch)
    ax3.tick_params(axis="both", length=2, width=1, labelsize=14)
    ax3.set_xlabel("Epoch", fontsize=22)
    ax3.set_ylabel("Test F1 Accuracy", fontsize=22)
    ax3.set_title("Test F1 vs Epoch", fontsize=32)
    plt.savefig(os.path.join("./data/" ,"task_test_f1_vs_epoch_%d.png" % args.model_no))
    
    return net