import random
from collections import defaultdict
from copy import deepcopy
from math import ceil

import nltk
import pandas as pd
import torch
from nltk import sent_tokenize
from statsmodels.tsa.statespace.tests.results.results_var_R import res_c
from transformers import AutoTokenizer
# import os
# print(os.environ['PYTHONPATH'])
from distance_utiles import eval_similarity, clac_dist_all_vecs, clac_dist_greedy_alignment, eval_K_best_similarity, \
    clac_dist_alignment, clac_dist_avg, eval_joined_similarity, get_close_vec, eval_similarity_vec, \
    clac_dist_median_vecs, calc_dist_ECDF, clac_dist_cosine_all_vecs, get_points, get_far_points, eval_ndcg
from imports import get_trained_model
from sentence_similarety import padded_tensor
from utils import get_data_with_index
import numpy as np

import matplotlib.pyplot as plt

from tfidf import avrg_idf, get_idf

seed_val = 1234
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

def get_common_word(x):
    d = defaultdict(int)
    for phrase in x['similars'].split('\n')[:5]:
        for w in phrase.split('_'):
            d[w] += 1
    word = sorted(d.items(), key=lambda x:x[1], reverse=True)[0][0]
    return word

def get_edge(x):
    return tuple(sorted([x['word'], x['choose']]))

def padded_tensor(arr):
    b = np.zeros([len(arr), len(max(arr, key=lambda x: len(x)))])
    for i, j in enumerate(arr):
        b[i][0:len(j)] = j
    return torch.tensor(b,dtype=torch.long).cuda()

def get_gold(data):
    d = defaultdict(list)
    for word, similar in zip(data['word'], data['choose']):
        d[word.replace('_', ' ')].append(similar.replace('_', ' '))
    l = list(d.items())
    for k, v in l:
        if v.count('0') + v.count('1') >=3:
            del d[k]
    return d
def get_indexes(words, indexes):
    if not all(w in indexes for w in words):
        return set()

    cur_set = set(indexes[words[0]])
    for w in words:
        cur_set = set.intersection(cur_set, set(indexes[w]))
    return cur_set


def filter_idf(final_sents, idf, p):
    sents = [(avrg_idf(s, idf), s) for s in final_sents]
    sents = sorted(sents, reverse=True)
    # length = int(len(final_sents)*p)
    length = p
    return [s[1] for s in sents[:length]]


def preproccess(sents, phrases):
    if not sents:
        return sents
    sents = set(sents)
    # return sents
    final_sents = []
    for sen in sents:
        for s in sent_tokenize(sen):
            if phrases in s:
                final_sents.append(s)
    return final_sents

def get_sentence_embedding_AUX(phrases, index, data, idf=None, p=1):
    sent_dict = {}
    for phrase in phrases:
        words = nltk.word_tokenize(phrase)
        indx = get_indexes(words, index)
        final_sents = [data[x] for x in indx]
        final_sents = preproccess(final_sents, phrase)
        if idf:
            final_sents = filter_idf(final_sents, idf, p)
        sents_embd = embd(final_sents, phrase)
        sent_dict[phrase] = sents_embd
    return sent_dict
def get_sentence_embedding(phrases, data_filename):
    index, data = get_data_with_index(data_filename, number=200000)
    return get_sentence_embedding_AUX(phrases, index, data)

def prepare_sentence(sents, phrase):
    sents = list(set(sents))
    final_sents = []
    for sen in sents:
        parts = nltk.sent_tokenize(sen)
        for i, s in enumerate(parts):
            if phrase not in s:
                continue
            final_sents.append(s)
    final_sents = set(final_sents)
    final_sents = [x.replace(phrase, '[MASK]') for x in final_sents]
    return final_sents


def embd(final_sents, phrase):
    mask_id = tokenizer.convert_tokens_to_ids('[MASK]')
    batch_size = 64
    sents = []
    final_sents = prepare_sentence(final_sents, phrase)
    for i in range(ceil(len(final_sents) / batch_size)):
        cur_sent = final_sents[batch_size * i:batch_size * (i + 1)]
        encoded_input = tokenizer(cur_sent)
        # Compute token embeddings
        index = []
        encoded_input['input_ids'] = [x for x in encoded_input['input_ids'] if len(x) < 500]  # remove sentences that are too long for bert
        for sen in encoded_input['input_ids']:
            index.append([i for i,x in enumerate(sen) if x == mask_id])
        index = [x if len(x) == 1 else [x[0]] for x in index]
        index = torch.tensor(index).cuda()
        # encoded_input = encoded_input.to('cuda')
        with torch.no_grad():
            # print(max(len(x) for x in encoded_input['input_ids']))
            model_output = model.forward_aux(padded_tensor(encoded_input['input_ids']), torch.tensor(index).squeeze().to('cuda')).detach()

        sents.append(model_output)
    if len(sents) > 1:
        return torch.cat(sents, dim=0)
    elif len(sents) == 1:
        return sents[0]
    return torch.zeros((1, 28996))
    # total_embd = torch.cat(sents, dim=0)
    # return (total_embd.sum(0).cpu() / len(final_sents)).numpy()


def remove_zeroes(embeddings):
    for k in [x for x in embeddings if torch.sum(embeddings[x]) == 0]:
        del embeddings[k]
    return embeddings

def get_word_embding():
    path = r"D:\glove\glove.6B.200d.txt"
    embeddings_dict = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict


def get_word_embeddings(all_nps, glove):
    res = defaultdict(list)
    for np in all_nps:
        for n in np.split(' '):
            if n in glove:
                res[np].append(torch.tensor(glove[n]))
            else:
                print(f'"{n}" is not ion vocab!')
    for k, v in res.items():
        res[k] = torch.stack(v)
    return res

fix_list = lambda x: list(filter(None, x))

def eval_human(gold):
    good = 0
    for g in gold:
        l = deepcopy(gold[g])
        i = l[-1]
        l = l[:-1]
        if i in l:
            good +=1
    return good/len(gold)

# def draw_data(e1, e2, ns):
#     for noun in ns:
#         cur_data = data[data['common'] == noun]
#         all_nps = cur_data.iloc[0]['similars'].split('\n') + [cur_data.iloc[0]['word']]
#         all_nps = [x.replace('_', ' ') for x in all_nps]
#         gold = get_gold(cur_data)
#         embeddings = get_sentence_embedding(all_nps, 'Arts_Crafts_and_Sewing_5.json.gz')
#         embeddings = remove_zeroes(embeddings)
#         word_embeddings = get_word_embeddings(all_nps, glove)

def add_plot(points, positive):
    ps, labels = list(zip(*points))
    ps_split =  list(zip(*ps))
    if positive:
        plt.plot(ps_split[0], ps_split[1], '+', color='blue')
    else:
        plt.plot(ps_split[0], ps_split[1], 'o', color='red')
    for i, txt in enumerate(labels):
        plt.annotate(txt, (ps_split[0][i], ps_split[1][i]))
def plot_points(good_l, bad_l, names):
    for good, bad, name in zip(good_l, bad_l, names):
        add_plot(good, True)
        add_plot(bad, False)
        # good_p = list(zip(*good))
        # bad_p = list(zip(*bad))
        # plt.plot(bad_p[0], bad_p[1], 'o', color='red')
        # plt.plot(good_p[0], good_p[1], '+', color='blue')
        plt.title(name)
        plt.savefig('imgs/{}.jpg'.format(name.replace(' ', '_')))
        plt.clf()
    # plt.show()


def print_csv(d, path):
    df = pd.DataFrame(d)
    df.sort_values(by='dist_difference', ascending=False)
    df.to_csv(path)


def calc_similarity():
    res_words = []
    res_context = []
    res_tot = []
    for noun in ns:
        # print(f'****{noun}****')
        cur_data = data[data['common'] == noun]
        all_nps = cur_data.iloc[0]['similars'].split('\n') + [cur_data.iloc[0]['word']]
        all_nps = [x.replace('_', ' ') for x in all_nps]
        gold = get_gold(cur_data)
        # embeddings = get_sentence_embedding(all_nps, raw_path)
        embeddings = get_sentence_embedding_AUX(all_nps, index, docs, idf, p)
        # embeddings = get_sentence_embedding_AUX(all_nps, index, docs)
        embeddings = remove_zeroes(embeddings)
        word_embeddings = get_word_embeddings(all_nps, glove)
        acc_words = 0  # eval_similarity(gold, word_embeddings, clac_dist_cosine_all_vecs)
        # acc_words = eval_human(gold)
        acc_context = eval_similarity(gold, embeddings, calc_dist_ECDF)

        res_words.append(acc_words)
        res_context.append(acc_context)
        # res_tot.append(max(acc_words, acc_context))
        res_words = fix_list(res_words)
        res_context = fix_list(res_context)
        res_tot = fix_list(res_tot)
        return res_words, res_context, res_tot

def old():

    res_words = []
    res_context = []
    res_tot = []
    for noun in ns:
        # print(f'****{noun}****')
        cur_data = data[data['common'] == noun]
        all_nps = cur_data.iloc[0]['similars'].split('\n') + [cur_data.iloc[0]['word']]
        all_nps = [x.replace('_', ' ') for x in all_nps]
        gold = get_gold(cur_data)
        # embeddings = get_sentence_embedding(all_nps, raw_path)
        # embeddings = get_sentence_embedding_AUX(all_nps, index, docs, idf, p)
        embeddings = get_sentence_embedding_AUX(all_nps, index, docs)
        embeddings = remove_zeroes(embeddings)
        word_embeddings = get_word_embeddings(all_nps, glove)
        acc_words = 0  # eval_similarity(gold, word_embeddings, clac_dist_cosine_all_vecs)
        # acc_words = eval_human(gold)
        acc_context = eval_similarity(gold, embeddings, calc_dist_ECDF)
        # acc_context = eval_similarity_vec(gold, embeddings)
        # acc_combined = eval_joined_similarity(gold, embeddings, word_embeddings, calc_dist_ECDF, clac_dist_cosine_all_vecs, 0.5)
        # ce, we = eval_ndcg(gold, sims, embeddings, word_embeddings, calc_dist_ECDF, clac_dist_cosine_all_vecs)
        # ce_tot.extend(ce)
        # we_tot.extend(we)
        ####
        # names, good, bad = get_points(gold, embeddings, word_embeddings, calc_dist_ECDF, clac_dist_cosine_all_vecs, 0.5)
        #########
        # contexts, words = get_far_points(gold, embeddings, word_embeddings, calc_dist_ECDF, clac_dist_cosine_all_vecs, 0.5)
        # all_contexts.extend(contexts)
        # all_words.extend(words)
        #####

        # plot_points(good, bad, names)
        # acc_combined = eval_K_best_similarity(gold, word_embeddings, embeddings, clac_dist_all_vecs, clac_dist_greedy_alignment, 3)
        # res_tot.append(acc_combined)

        # print(f'acc for {noun}')
        # print(f'words: {acc_words}, context: {acc_context}, combined: {acc_combined}')
        # if acc_context >= acc_words:
        #     print('context is better!')
        res_words.append(acc_words)
        res_context.append(acc_context)
        # res_tot.append(max(acc_words, acc_context))
        res_words = fix_list(res_words)
        res_context = fix_list(res_context)
        res_tot = fix_list(res_tot)
    total.append((p, np.mean(res_context)))
    print(total)

def calc_similarity(ns):
    res_words = []
    res_context = []
    res_tot = []
    for noun in ns:
        # print(f'****{noun}****')
        cur_data = data[data['common'] == noun]
        all_nps = cur_data.iloc[0]['similars'].split('\n') + [cur_data.iloc[0]['word']]
        all_nps = [x.replace('_', ' ') for x in all_nps]
        gold = get_gold(cur_data)
        # embeddings = get_sentence_embedding_AUX(all_nps, index, docs, idf, p)
        embeddings = get_sentence_embedding_AUX(all_nps, index, docs)
        embeddings = remove_zeroes(embeddings)
        word_embeddings = get_word_embeddings(all_nps, glove)
        acc_words = eval_similarity(gold, word_embeddings, clac_dist_cosine_all_vecs)
        # acc_words = eval_human(gold)
        acc_context = eval_similarity(gold, embeddings, calc_dist_ECDF)

        res_words.append(acc_words)
        res_context.append(acc_context)
        # res_tot.append(max(acc_words, acc_context))
        res_words = fix_list(res_words)
        res_context = fix_list(res_context)
        res_tot = fix_list(res_tot)
    return res_words, res_context, res_tot

if __name__ == '__main__':
    model_name = 'SpanBERT/spanbert-base-cased'
    data_path = r"D:\human\cell1500.csv"
    # data_path = r"D:\human\art1500.csv"
    data_path = r"D:\human\automotive1500.csv"

    # raw_path = 'All_Beauty_5.json.gz'
    # raw_path = 'Cell_Phones_and_Accessories_5.json.gz'
    raw_path = 'Automotive_5.json.gz'
    # raw_path = 'Arts_Crafts_and_Sewing_5.json.gz'
    # clusters_path = r"D:\human\c_cell.json"
    clusters_path = r"D:\human\c_art.json"
    model = get_trained_model(model_name, path="./data/")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # data = pd.read_csv(r"D:\human\f1811941.csv")
    # data = pd.read_csv(r"D:\human\art1500.csv")
    data = pd.read_csv(data_path)
    # data = pd.read_csv(r"D:\human\automotive1500.csv")

    data['common'] = data.apply(get_common_word, axis=1)
    # print(data['common'])

    ns = sorted(list(set(data['common'].to_list())))
    glove = get_word_embding()
    alphas = np.arange(0,1, 0.1)
    values = []
    # for alpha in alphas:
    count = 0
    all_contexts = []
    all_words = []
    # sims = gen_similarities(clusters_path)
    ce_tot = []
    we_tot = []
    index, docs = get_data_with_index(raw_path, number=200000)
    idf = get_idf(raw_path, docs)

    total = []
    # for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    # for p in range(10,100,10):
    res_words, res_context, res_tot = calc_similarity(ns)
    print(np.mean(res_words))
    print(np.mean(res_context))

    # old()
    # print_csv(all_contexts, 'contexts.csv')
    # print_csv(all_words, 'words.csv')
    print(total)
    # print(f'overall word acc:{np.mean(res_words)}, context: {np.mean(res_context)}, aggregate: {np.mean(res_tot)}')
    # print(f'we ndcg: {np.mean(we_tot)}, ce ndcg: {np.mean(ce_tot)}')
    # print(ce_tot)
    # print(we_tot)
    # values.append(np.mean(res_tot))
    # print(values)

    # plt.plot(alphas, values, 'o')
    # plt.show()