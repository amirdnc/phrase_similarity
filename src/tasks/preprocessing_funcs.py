#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:12:22 2019

@author: weetee
"""
import os
import re
import random
import copy
import time

import pandas
import pandas as pd
import json
import torch
from dask import compute
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from ..misc import save_as_pickle, load_pickle
from tqdm import tqdm
import logging
import dask.dataframe as dd
from dask.multiprocessing import get
tqdm.pandas(desc="prog_bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def process_text(text, mode='train'):
    sents, relations, comments, blanks = [], [], [], []
    for i in range(int(len(text)/4)):
        sent = text[4*i]
        relation = text[4*i + 1]
        comment = text[4*i + 2]
        blank = text[4*i + 3]
        
        # check entries
        if mode == 'train':
            assert int(re.match("^\d+", sent)[0]) == (i + 1)
        else:
            assert (int(re.match("^\d+", sent)[0]) - 8000) == (i + 1)
        assert re.match("^Comment", comment)
        assert len(blank) == 1
        
        sent = re.findall("\"(.+)\"", sent)[0]
        sent = re.sub('<e1>', '[E1]', sent)
        sent = re.sub('</e1>', '[/E1]', sent)
        sent = re.sub('<e2>', '[E2]', sent)
        sent = re.sub('</e2>', '[/E2]', sent)
        sents.append(sent); relations.append(relation), comments.append(comment); blanks.append(blank)
    return sents, relations, comments, blanks


def getn_indx(sen):
    pass

def get_noun_df(text):
    pos0 = [x['pos'][0] for x in text]
    pos1 = [x['pos'][1] for x in text]
    neg = [x['neg'] for x in text]
    return pd.DataFrame(data={'pos0': pos0, 'pos1': pos1, 'neg': neg})

def get_multi_noun_df(text):
    # text = text[:4023]  # for debuging
    random.shuffle(text)
    return pd.DataFrame.from_records(text)  # .transpose()
    pos = [x['pos'] for x in text]
    neg = [x['neg'] for x in text]
    if 'neg2' not in text[0]:  #support old format
        return pd.DataFrame(data={'pos': pos, 'neg': neg})
    neg2 = [x['neg2'] for x in text]
    # return pd.DataFrame(data={'pos': pos, 'neg': neg, 'neg2': neg2})



def preprocess_noun(args):
    data_path = args.train_data  # './data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    logger.info("Reading training file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        text_train = json.load(f)

    with open(args.test_data, 'r', encoding='utf8') as f:
        text_test = json.load(f)
    return get_noun_df(text_train), get_noun_df(text_test)

def preprocess_multi_noun(args):
    data_path = args.train_data  # './data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    logger.info("Reading training file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        text_train = json.load(f)

    with open(args.test_data, 'r', encoding='utf8') as f:
        text_test = json.load(f)
    # text_train = text_train[40000: 100000]
    text_test = text_test[: 5000]
    return get_multi_noun_df(text_train), get_multi_noun_df(text_test)
    # return get_multi_noun_df(text_train[:1000]), get_multi_noun_df(text_test[:300])

def preprocess_semeval2010_8(args):
    '''
    Data preprocessing for SemEval2010 task 8 dataset
    '''
    data_path = args.train_data #'./data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    logger.info("Reading training file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        text = f.readlines()
    
    sents, relations, comments, blanks = process_text(text, 'train')
    df_train = pd.DataFrame(data={'sents': sents, 'relations': relations})
    
    data_path = args.test_data #'./data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
    logger.info("Reading test file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        text = f.readlines()
    
    sents, relations, comments, blanks = process_text(text, 'test')
    df_test = pd.DataFrame(data={'sents': sents, 'relations': relations})
    
    rm = Relations_Mapper(df_train['relations'])
    save_as_pickle('relations.pkl', rm)
    df_test['relations_id'] = df_test.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
    df_train['relations_id'] = df_train.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
    save_as_pickle('df_train.pkl', df_train)
    save_as_pickle('df_test.pkl', df_test)
    logger.info("Finished and saved!")
    
    return df_train, df_test, rm

class Relations_Mapper(object):
    def __init__(self, relations):
        self.rel2idx = {}
        self.idx2rel = {}
        
        logger.info("Mapping relations to IDs...")
        self.n_classes = 0
        for relation in tqdm(relations):
            if relation not in self.rel2idx.keys():
                self.rel2idx[relation] = self.n_classes
                self.n_classes += 1
        
        for key, value in self.rel2idx.items():
            self.idx2rel[value] = key


class Pad_Sequence():
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """

    def __init__(self, seq_pad_value, label_pad_value=-1, label2_pad_value=-1, \
                 ):
        self.seq_pad_value = seq_pad_value
        self.label_pad_value = label_pad_value
        self.label2_pad_value = label2_pad_value

    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        seqs = [x[0] for x in sorted_batch]
        seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=self.seq_pad_value)
        x_lengths = torch.LongTensor([len(x) for x in seqs])

        labels = list(map(lambda x: x[1], sorted_batch))
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.label_pad_value)
        y_lengths = torch.LongTensor([len(x) for x in labels])

        labels2 = list(map(lambda x: x[2], sorted_batch))
        labels2_padded = pad_sequence(labels2, batch_first=True, padding_value=self.label2_pad_value)
        y2_lengths = torch.LongTensor([len(x) for x in labels2])

        return seqs_padded, labels_padded, labels2_padded, \
               x_lengths, y_lengths, y2_lengths



class Pad_Sequence_Noun():
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """

    def __init__(self, seq_pad_value):
        self.seq_pad_value = seq_pad_value

    def pad_seq(self, batch, index):
        seq = [x[index] for x in batch]
        return pad_sequence(seq, batch_first=True, padding_value=self.seq_pad_value)
    def __call__(self, batch):
        masks = self.pad_seq(batch, 3)
        return self.pad_seq(batch, 0), self.pad_seq(batch, 1), self.pad_seq(batch, 2),  (masks[:,0], masks[:,1], masks[:,2])

class Pad_Sequence_Noun():
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """

    def __init__(self, seq_pad_value):
        self.seq_pad_value = seq_pad_value

    def pad_seq(self, batch, index):
        seq = [x[index] for x in batch]
        vecs = [pad_sequence([torch.LongTensor(x[i]) for x in seq], batch_first=True, padding_value=self.seq_pad_value) for i in range(len(seq[0]))]
        # for i in len(seq[0]):

        return vecs
    def __call__(self, batch):
        # masks = self.pad_seq(batch, 2)
        return [self.pad_seq(batch, x) for x in range(len(batch[0]))]
        # return self.pad_seq(batch, 0), self.pad_seq(batch, 1), self.pad_seq(batch, 2)

def get_e1e2_start(x, e1_id, e2_id):
    try:
        e1_e2_start = ([i for i, e in enumerate(x) if e == e1_id][0],\
                        [i for i, e in enumerate(x) if e == e2_id][0])
    except Exception as e:
        e1_e2_start = None
        print(e)
    return e1_e2_start

def get_mask_index(x, mask_id):
    try:
        mask_index = [i for i, e in enumerate(x) if e == mask_id]
        if mask_index:
            mask_index = mask_index[0]
        else:
            print('what?')
    except Exception as e:
        mask_index = None
        print(e)
    return mask_index

def tokenize_with_span(tokenizer, l, phrase, mask_id, infrence=False):
    phrase = tokenizer.encode(phrase)[1:-1]
    sents = []
    ii = []
    for index, s in enumerate(l):
        sent = tokenizer.encode(s)
        i = sent.index(mask_id)
        if False and (index <= (len(l)/3 - 1) or infrence): # testing
            sent = sent[:i] + phrase + sent[i+1:]
            length = (len(phrase))
        else:
            length = 1
        sents.append(sent)
        ii.append((i, length))
    return  (ii, sents)

def tokenize_df(tokenizer, df, sent_type):
    return df.progress_apply(lambda x: [tokenizer.encode(y.replace('[mask]', '[MASK]'))for y in x[sent_type]], axis=1)
    # ddata = dd.from_pandas(df, npartitions=30)
    # return ddata.map_partitions(
    #     lambda x: [tokenizer.encode(y.replace('[mask]', '[MASK]')) for y in x[sent_type]]).compute(scheduler='threads')

def tokenize_span(tokenizer, df, sent_type):
    if sent_type == 'neg2': #it's late, this works...
        new_df = df.progress_apply(lambda x: tokenize_with_span(tokenizer, x[sent_type], x['neg_phrase2'],
                                                                tokenizer.convert_tokens_to_ids('[MASK]')), axis=1)
    else:
        new_df = df.progress_apply(lambda x: tokenize_with_span(tokenizer, x[sent_type], x[sent_type + '_phrase'],
                                                          tokenizer.convert_tokens_to_ids('[MASK]')), axis=1)
    new_df = pd.DataFrame(new_df.to_list())
    new_df.rename(columns={0: 'mask_index', 1: 'tokens'}, inplace=True)
    return new_df

class multi_noun_dataset(Dataset):
    def __init__(self, df, tokenizer=None):
        if not tokenizer:
            self.df = df
            return
        mask_id = tokenizer.convert_tokens_to_ids('[MASK]')
        self.df = df
        self.df['pos_tokens'] = tokenize_df(tokenizer, df, 'pos')
        self.df['neg_tokens'] = tokenize_df(tokenizer, df, 'neg')
        print('start tokenize')
        t = time.time()
        p = self.df.progress_apply(lambda x: [get_mask_index(y, mask_id) for y in x['pos_tokens']], axis=1)
        n = self.df.progress_apply(lambda x: [get_mask_index(y, mask_id) for y in x['neg_tokens']], axis=1)
        print('finish tokenizing. took: {}'.format(time.time() - t))
        if 'neg2' in df:
            self.df['neg_tokens2'] = tokenize_df(tokenizer, df, 'neg2')
            n2 = self.df.progress_apply(lambda x: [get_mask_index(y, mask_id) for y in x['neg_tokens2']], axis=1)
            self.df['mask_index'] = [x for x in zip(p, n, n2)]
        else:
            self.df['mask_index'] = [x for x in zip(p, n)]
    def __len__(self, ):
        return len(self.df)

    def __getitem__(self, idx):
        if 'neg2' in self.df:
            return [(x) for x in self.df.iloc[idx]['pos_tokens']], \
                   [(x) for x in self.df.iloc[idx]['neg_tokens']], \
                   [(x) for x in self.df.iloc[idx]['neg_tokens2']], \
                   [(x) for x in self.df.iloc[idx]['mask_index']],
        else:
            return [(x) for x in self.df.iloc[idx]['pos_tokens']], \
                   [(x) for x in self.df.iloc[idx]['neg_tokens']], \
                   [(x) for x in self.df.iloc[idx]['mask_index']],


class span_dataset(Dataset):

    def __init__(self, df, tokenizer=None):
        if not tokenizer:
            self.df = df
            return
        mask_id = tokenizer.convert_tokens_to_ids('[MASK]')
        self.df = df
        tp = tokenize_span(tokenizer, df, 'pos')
        p = tp['mask_index']
        self.df['pos_tokens'] = tp['tokens']

        tn = tokenize_span(tokenizer, df, 'neg')
        n = tn['mask_index']
        self.df['neg_tokens'] = tn['tokens']

        tn2 = tokenize_span(tokenizer, df, 'neg2')
        n2 = tn2['mask_index']
        self.df['neg_tokens2'] = tn2['tokens']
        self.df['mask_index'] = [x for x in zip(p, n, n2)]
        # print('start tokenize')
        # t = time.time()
        # n = self.df.progress_apply(lambda x: [get_mask_index(y, mask_id) for y in x['neg_tokens']], axis=1)
        # print('finish tokenizing. took: {}'.format(time.time() - t))
        # if 'neg2' in df:
        #     self.df['neg_tokens2'] = tokenize_span(tokenizer, df, 'neg2')
        #     n2 = self.df.progress_apply(lambda x: [get_mask_index(y, mask_id) for y in x['neg_tokens2']], axis=1)
        #     self.df['mask_index'] = [x for x in zip(p, n, n2)]
        # else:
        #     self.df['mask_index'] = [x for x in zip(p, n)]
    def __len__(self, ):
        return len(self.df)

    def __getitem__(self, idx):
        return [(x) for x in self.df.iloc[idx]['pos_tokens']], \
               [(x) for x in self.df.iloc[idx]['neg_tokens']], \
               [(x) for x in self.df.iloc[idx]['neg_tokens2']], \
               [(x) for x in self.df.iloc[idx]['mask_index']],
class noun_dataset(Dataset):
    def __init__(self, df, tokenizer):
        mask_id = tokenizer.convert_tokens_to_ids('[MASK]')
        self.df = df
        self.df['pos0_tokens'] = self.df.progress_apply(lambda x: tokenizer.encode(x['pos0'].replace('[mask]', '[MASK]')), axis=1)
        self.df['pos1_tokens'] = self.df.progress_apply(lambda x: tokenizer.encode(x['pos1'].replace('[mask]', '[MASK]')), axis=1)
        self.df['neg_tokens'] = self.df.progress_apply(lambda x: tokenizer.encode(x['neg'].replace('[mask]', '[MASK]')), axis=1)
        p0 = self.df.progress_apply(lambda x: get_mask_index(x['pos0_tokens'], mask_id), axis=1)
        p1 = self.df.progress_apply(lambda x: get_mask_index(x['pos1_tokens'], mask_id), axis=1)
        n = self.df.progress_apply(lambda x: get_mask_index(x['neg_tokens'], mask_id), axis=1)
        self.df['mask_index'] = [x for x in zip(p0, p1, n)]

    def __len__(self, ):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.LongTensor(self.df.iloc[idx]['pos0_tokens']), \
               torch.LongTensor(self.df.iloc[idx]['pos1_tokens']), \
               torch.LongTensor(self.df.iloc[idx]['neg_tokens']), \
               torch.LongTensor(self.df.iloc[idx]['mask_index'])

class semeval_dataset(Dataset):
    def __init__(self, df, tokenizer, e1_id, e2_id):
        self.e1_id = e1_id
        self.e2_id = e2_id
        self.df = df
        logger.info("Tokenizing data...")
        self.df['input'] = self.df.progress_apply(lambda x: tokenizer.encode(x['sents']),\
                                                             axis=1)
        
        self.df['e1_e2_start'] = self.df.progress_apply(lambda x: get_e1e2_start(x['input'],\
                                                       e1_id=self.e1_id, e2_id=self.e2_id), axis=1)
        print("\nInvalid rows/total: %d/%d" % (df['e1_e2_start'].isnull().sum(), len(df)))
        self.df.dropna(axis=0, inplace=True)
    
    def __len__(self,):
        return len(self.df)
        
    def __getitem__(self, idx):
        return torch.LongTensor(self.df.iloc[idx]['input']),\
                torch.LongTensor(self.df.iloc[idx]['e1_e2_start']),\
                torch.LongTensor([self.df.iloc[idx]['relations_id']])

def preprocess_fewrel(args, do_lower_case=True):
    '''
    train: train_wiki.json
    test: val_wiki.json
    For 5 way 1 shot
    '''
    def process_data(data_dict):
        sents = []
        labels = []
        for relation, dataset in data_dict.items():
            for data in dataset:
                # first, get & verify the positions of entities
                h_pos, t_pos = data['h'][-1], data['t'][-1]
                
                if not len(h_pos) == len(t_pos) == 1: # remove one-to-many relation mappings
                    continue
                
                h_pos, t_pos = h_pos[0], t_pos[0]
                
                if len(h_pos) > 1:
                    running_list = [i for i in range(min(h_pos), max(h_pos) + 1)]
                    assert h_pos == running_list
                    h_pos = [h_pos[0], h_pos[-1] + 1]
                else:
                    h_pos.append(h_pos[0] + 1)
                
                if len(t_pos) > 1:
                    running_list = [i for i in range(min(t_pos), max(t_pos) + 1)]
                    assert t_pos == running_list
                    t_pos = [t_pos[0], t_pos[-1] + 1]
                else:
                    t_pos.append(t_pos[0] + 1)
                
                if (t_pos[0] <= h_pos[-1] <= t_pos[-1]) or (h_pos[0] <= t_pos[-1] <= h_pos[-1]): # remove entities not separated by at least one token 
                    continue
                
                if do_lower_case:
                    data['tokens'] = [token.lower() for token in data['tokens']]
                
                # add entity markers
                if h_pos[-1] < t_pos[0]:
                    tokens = data['tokens'][:h_pos[0]] + ['[E1]'] + data['tokens'][h_pos[0]:h_pos[1]] \
                            + ['[/E1]'] + data['tokens'][h_pos[1]:t_pos[0]] + ['[E2]'] + \
                            data['tokens'][t_pos[0]:t_pos[1]] + ['[/E2]'] + data['tokens'][t_pos[1]:]
                else:
                    tokens = data['tokens'][:t_pos[0]] + ['[E2]'] + data['tokens'][t_pos[0]:t_pos[1]] \
                            + ['[/E2]'] + data['tokens'][t_pos[1]:h_pos[0]] + ['[E1]'] + \
                            data['tokens'][h_pos[0]:h_pos[1]] + ['[/E1]'] + data['tokens'][h_pos[1]:]
                
                assert len(tokens) == (len(data['tokens']) + 4)
                sents.append(tokens)
                labels.append(relation)
        return sents, labels
        
    with open('./data/fewrel/train_wiki.json') as f:
        train_data = json.load(f)
        
    with  open('./data/fewrel/val_wiki.json') as f:
        test_data = json.load(f)
    
    train_sents, train_labels = process_data(train_data)
    test_sents, test_labels = process_data(test_data)
    
    df_train = pd.DataFrame(data={'sents': train_sents, 'labels': train_labels})
    df_test = pd.DataFrame(data={'sents': test_sents, 'labels': test_labels})
    
    rm = Relations_Mapper(list(df_train['labels'].unique()))
    save_as_pickle('relations.pkl', rm)
    df_train['labels'] = df_train.progress_apply(lambda x: rm.rel2idx[x['labels']], axis=1)
    
    return df_train, df_test

class fewrel_dataset(Dataset):
    def __init__(self, df, tokenizer, seq_pad_value, e1_id, e2_id):
        self.e1_id = e1_id
        self.e2_id = e2_id
        self.N = 5
        self.K = 1
        self.df = df
        
        logger.info("Tokenizing data...")
        self.df['sents'] = self.df.progress_apply(lambda x: tokenizer.encode(" ".join(x['sents'])),\
                                      axis=1)
        self.df['e1_e2_start'] = self.df.progress_apply(lambda x: get_e1e2_start(x['sents'],\
                                                       e1_id=self.e1_id, e2_id=self.e2_id), axis=1)
        print("\nInvalid rows/total: %d/%d" % (self.df['e1_e2_start'].isnull().sum(), len(self.df)))
        self.df.dropna(axis=0, inplace=True)
        
        self.relations = list(self.df['labels'].unique())
        
        self.seq_pad_value = seq_pad_value
            
    def __len__(self,):
        return len(self.df)
    
    def __getitem__(self, idx):
        target_relation = self.df['labels'].iloc[idx]
        relations_pool = copy.deepcopy(self.relations)
        relations_pool.remove(target_relation)
        sampled_relation = random.sample(relations_pool, self.N - 1)
        sampled_relation.append(target_relation)
        
        target_idx = self.N - 1
    
        e1_e2_start = []
        meta_train_input, meta_train_labels = [], []
        for sample_idx, r in enumerate(sampled_relation):
            filtered_samples = self.df[self.df['labels'] == r][['sents', 'e1_e2_start', 'labels']]
            sampled_idxs = random.sample(list(i for i in range(len(filtered_samples))), self.K)
            
            sampled_sents, sampled_e1_e2_starts = [], []
            for sampled_idx in sampled_idxs:
                sampled_sent = filtered_samples['sents'].iloc[sampled_idx]
                sampled_e1_e2_start = filtered_samples['e1_e2_start'].iloc[sampled_idx]
                
                assert filtered_samples['labels'].iloc[sampled_idx] == r
                
                sampled_sents.append(sampled_sent)
                sampled_e1_e2_starts.append(sampled_e1_e2_start)
            
            meta_train_input.append(torch.LongTensor(sampled_sents).squeeze())
            e1_e2_start.append(sampled_e1_e2_starts[0])
            
            meta_train_labels.append([sample_idx])
            
        meta_test_input = self.df['sents'].iloc[idx]
        meta_test_labels = [target_idx]
        
        e1_e2_start.append(get_e1e2_start(meta_test_input, e1_id=self.e1_id, e2_id=self.e2_id))
        e1_e2_start = torch.LongTensor(e1_e2_start).squeeze()
        
        meta_input = meta_train_input + [torch.LongTensor(meta_test_input)]
        meta_labels = meta_train_labels + [meta_test_labels]
        meta_input_padded = pad_sequence(meta_input, batch_first=True, padding_value=self.seq_pad_value).squeeze()
        return meta_input_padded, e1_e2_start, torch.LongTensor(meta_labels).squeeze()

def load_dataloaders(args, test_only=False):

    # if args.model_no == 3:
    model = args.model
    lower_case = True
    model_name = args.model

    if os.path.isfile("./data/%s_tokenizer.pkl" % model_name):
        tokenizer = load_pickle("%s_tokenizer.pkl" % model_name)
        logger.info("Loaded tokenizer from pre-trained blanks model")
    else:
        logger.info("Pre-trained blanks tokenizer not found, initializing new tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(args.model)

        save_as_pickle("%s_tokenizer.pkl" % model_name.replace('/', '_'), tokenizer, args.save_path)
        logger.info("Saved %s tokenizer at ./data/%s_tokenizer.pkl" %(model_name, model_name))

    PS = Pad_Sequence_Noun(seq_pad_value=tokenizer.pad_token_id)

    path_train = '{}_{}.pkl'.format(args.task, os.path.basename(args.train_data))
    path_test = '{}_{}.pkl'.format(args.task, os.path.basename(args.test_data))
    if args.task == 'noun_similarity':
        df_train, df_test = preprocess_noun(args)
        train_set = noun_dataset(df_train, tokenizer=tokenizer)
        test_set = noun_dataset(df_test, tokenizer=tokenizer)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, \
                                  num_workers=0, collate_fn=PS, pin_memory=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, \
                                 num_workers=0, collate_fn=PS, pin_memory=False)

    elif args.task == 'multi_noun_similarity':

        if os.path.exists(path_train):
            test_set = multi_noun_dataset(pandas.read_pickle(path_test))
            train_set = multi_noun_dataset(pandas.read_pickle(path_train))
            train_loader = DataLoader(train_set, batch_size=args.batch_size * torch.cuda.device_count(), shuffle=True, \
                                      num_workers=0, collate_fn=PS, pin_memory=False)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, \
                                     num_workers=0, collate_fn=PS, pin_memory=False)
            return train_loader, test_loader, len(train_loader), len(test_loader)
        else:
            df_train, df_test = preprocess_multi_noun(args)
            # df_test.to_csv(test_path)
            # df_test.to_csv(train_path)
        if not test_only:
            train_set = multi_noun_dataset(df_train, tokenizer=tokenizer)
            print('batch size is {} * {} = {}'.format(torch.cuda.device_count(), args.batch_size, torch.cuda.device_count()* args.batch_size ))
            train_loader = DataLoader(train_set, batch_size=args.batch_size * torch.cuda.device_count(), shuffle=True, \
                                      num_workers=0, collate_fn=PS, pin_memory=False)
        else:
            train_loader = None
        # if args.task == 'double_negative_similarity':
        test_set = multi_noun_dataset(df_test, tokenizer=tokenizer)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, \
                                 num_workers=0, collate_fn=PS, pin_memory=False)
        if not os.path.exists(path_train):
            with open(path_train, 'wb') as f:
                train_set.df.to_pickle(f)

            with open(path_test, 'wb') as f:
                test_set.df.to_pickle(f)
    elif args.task == 'double_negative_similarity':

        if os.path.exists(path_train):
            test_set = span_dataset(pandas.read_pickle(path_test))
            train_set = span_dataset(pandas.read_pickle(path_train))
            train_loader = DataLoader(train_set, batch_size=args.batch_size * torch.cuda.device_count(), shuffle=True, \
                                      num_workers=0, collate_fn=PS, pin_memory=False)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, \
                                     num_workers=0, collate_fn=PS, pin_memory=False)
            return train_loader, test_loader, len(train_loader), len(test_loader)
        if not test_only:

            df_train, df_test = preprocess_multi_noun(args)
            train_set = span_dataset(df_train, tokenizer=tokenizer)
            print('batch size is {} * {} = {}'.format(torch.cuda.device_count(), args.batch_size,
                                                      torch.cuda.device_count() * args.batch_size))
            train_loader = DataLoader(train_set, batch_size=args.batch_size * torch.cuda.device_count(), shuffle=True, \
                                      num_workers=0, collate_fn=PS, pin_memory=False)
            with open(path_train,'wb') as f:
                train_set.df.to_pickle(f)
        else:
            train_loader = None
        test_set = span_dataset(df_test, tokenizer=tokenizer)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, \
                                 num_workers=0, collate_fn=PS, pin_memory=False)
        with open(path_test,'wb') as f:
            test_set.df.to_pickle(path_test)
    train_length = len(train_set)

    test_length = len(test_set)

    return train_loader, test_loader, train_length, test_length