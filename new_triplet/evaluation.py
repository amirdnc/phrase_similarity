import json
import os
import pickle
from collections import defaultdict

import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoTokenizer

from distance_utiles import calc_dist_ECDF, calc_cosine_ECDF, ECDF, calc_cosine_ECDF_knn
from imports import get_trained_model
from phrase_similarity import embd
from wikipedia_similarity import get_sentece_context, eval_list
from web.datasets.similarity import fetch_TR9856




def ppdb_embd_eval(data_path, model_name, embeddings, save_location, num_iterations=150, num_context=150, start_index=0, model=None, k=5):
    with open(data_path) as f:
        data = json.load(f)

    threshold = 0.057065253464028465
    # if not model:
    #     model = get_trained_model(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    res = []

    # model = SentenceTransformer('whaleloops/phrase-bert')
    pos = 0; neg = 0; n_d = defaultdict(list)
    # dist_calculator = ECDF(embeddings, model)

    good_save_location = save_location +'/good/'
    bad_save_location = save_location +'/bad/'
    if not os.path.exists(save_location):
        os.mkdir(save_location)
    if not os.path.exists(bad_save_location):
        os.mkdir(bad_save_location)
    if not os.path.exists(good_save_location):
        os.mkdir(good_save_location)

    for i, cur in enumerate(tqdm(data)):  #tqdm
        if i < start_index:
            continue
        p1, p2, label = cur
        if i > num_iterations+start_index:
            break

        int_label = 1 if label == 'pos' else 0
        # p1_context = get_sentece_context(p1)[:num_context]
        # p2_context = get_sentece_context(p2)[:num_context]
        # p1_lexical, p2_lexical = model.encode([p1, p2], batch_size=2, show_progress_bar=False)
        p1_context = embeddings[p1]
        p2_context = embeddings[p2]
        if len(p1_context) < 100 or len(p2_context) < 100:
            continue
        p1_embd = torch.tensor(embeddings[p1])
        p2_embd = torch.tensor(embeddings[p2])
        if model:
            p1_embd = model(p1_embd.cuda())
            p2_embd = model(p2_embd.cuda())
        # p1_augment = torch.cat([p1_embd, torch.tensor(p1_lexical).expand_as(p1_embd)], dim=1)
        # p2_augment = torch.cat([p2_embd, torch.tensor(p2_lexical).expand_as(p2_embd)], dim=1)
        # p1_augment = p1_embd + torch.tensor(p1_lexical).expand_as(p1_embd)
        # p2_augment = p2_embd+ torch.tensor(p2_lexical).expand_as(p2_embd)

        # dist = calc_dist_ECDF(p1_embd, p2_embd)

        # dist = calc_cosine_ECDF(p1_embd, p2_embd)
        dist = calc_cosine_ECDF_knn(p1_embd, p2_embd, k)
        # dist = calc_dist_ECDF(p1_embd, p2_embd)
        # dist = dist_calculator.calc_dist(p1_embd, p2_embd)
        # if dist < threshold and label == 'pos' or dist >= threshold and label == 'neg':
        #     draw_sample(p1, p2, p1_embd, p2_embd, dist, label, good_save_location)
        # else:
        #     draw_sample(p1, p2, p1_embd, p2_embd, dist, label, bad_save_location)
        new_save_location = save_location + f'/{p1} - {p2}/'
        # if not os.path.exists(new_save_location):
        #     os.mkdir(new_save_location)
        # for i2, i1 in enumerate(ind1):
        #     draw_sample(p1, p2, p1_embd, p2_embd, dist, label, new_save_location+f'{i2}_', ind1=i1, ind2=i2)
        res.append((dist, int_label))
        if i % 200 == 199:
            print(eval_list(res, False))
    print(f'k = {k}, res = {res[0]}')
    return eval_list(res, False)

if __name__ == '__main__':
    # fetch_TR9856()
    data_path = r'D:\vec_dict.pickle'
    # data_path = r'D:\vec_dict_tiny.pickle'
    with open(data_path, 'rb') as handle:
        data = pickle.load(handle)
    # print(ppdb_embd_eval("D:\PPDB-filtered.json", 'SpanBERT/spanbert-base-cased', data, 'temp/temp/', num_iterations=99999, start_index=11649,
    #                 num_context=99999))
    res = []
    for i in range(1,10):
        res.append(ppdb_embd_eval("D:\PPDB-filtered.json", 'SpanBERT/spanbert-base-cased', data, 'temp/temp/',
                       num_iterations=99999, start_index=11649, num_context=99999, k=i))
        print(f'results: {res}')