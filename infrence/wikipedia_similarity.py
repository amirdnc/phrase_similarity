import json
import os
import urllib
from collections import defaultdict
from io import StringIO

import requests
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from distance_utiles import get_close, calc_dist_ECDF, clac_dist_median_vecs, clac_dist_avg, clac_cosin, get_far, \
    clac_dot, calc_cosine_ECDF, calc_cosine_ECDF_knn
from imports import get_trained_model
from new_triplet.utils import draw_sample, get_model
from phrase_similarity import get_word_embding, embd
import matplotlib.pyplot as plt

import torch

import spacy
nlp = spacy.load('en_core_web_sm')
WIKIPEDIA_URL = "https://spike.staging.apps.allenai.org/api/3/multi-search/query" #  "https://spike.staging.apps.allenai.org/api/3/search/query"
WIKIPEDIA_URL = "https://spike.more-mobx-fixes.apps.allenai.org/api/3/multi-search/query"
WIKIPEDIA_BASE_URL = "https://spike.staging.apps.allenai.org"
WIKIPEDIA_BASE_URL = "https://spike.more-mobx-fixes.apps.allenai.org"
MAX_ROWS = 2000

def get_csv_from_url(url, max_lines):
    strings = []
    try:
        r = urllib.request.urlopen(url)
    except:
        r= urllib.request.urlopen(url)
    for index, line in enumerate(r):
        strings.append(line.decode('utf-8'))
        if index > max_lines:
            break
    raw = '\n'.join(strings)
    try:
        df = pd.read_csv(StringIO(raw))
    except:
        df = pd.read_csv(StringIO(raw),  engine='python', error_bad_lines=False)
    return df

def get_df_from_response(response, base_url):
    if response.status_code != 204:
        print('bad status code :{}'.format(response.status_code))
        exit()
    url = base_url + response.headers['csv-location'] + '?limit=2200'  # limit csv size
    # s = requests.get(url).content
    # reader = pd.read_csv(url, iterator=True)
    # c = reader.get_chunk(MAX_ROWS)
    # c = get_csv_from_url(url, MAX_ROWS)
    c = pd.read_csv(url)
    return c

def load_turney_data(data_path):
    turney_data_fname = data_path
    with open(turney_data_fname, 'r') as f:
        content = f.readlines()
        data_list = []
        for line in content:
            components = line.strip('\n').split(' | ')
            data_list.append(components)
    return data_list

def get_sentece_context(phrase, use_paragraphs=False, data_dir='D:\sentence_context/'):
        phrase_path = data_dir + phrase.replace(' ', '_') +'.csv'
        if os.path.exists(phrase_path):
            df = pd.read_csv(phrase_path)
        else:
            print(f'No context for {phrase}. Retriving from spike...')
            try:
                df = perform_query('`' + phrase + '`')
                df.to_csv(phrase_path)
            except:
                print(f'error when parsing {phrase}')
                return []
        if use_paragraphs:
            return df['paragraph_text'].to_list()
        if 'sentence_text' in df:
            return df['sentence_text'].to_list()
        elif'sentence' in df:
            return df['sentence'].to_list()
        else:
            return []

def context_to_dict(sentences):
    d = defaultdict(int)
    for s in sentences:
        doc = nlp(s)
        for token in doc:
            d[token.lemma_] += 1
    total = sum(d.values())
    d = {x:y/total for x,y in d.items()}
    return d

def dist_d(d1, d2):

    return sum([d2[val]*d1[val] for val in d1 if val in d2])

def get_far_d(embeddings, g):
    best_phrase = None
    best_dist = 0
    for cur in embeddings:
        if cur == g:
            continue
        # if cur == 'jigsaw':
        #     print('here')
        # print(cur)
        dist = dist_d(embeddings[g], embeddings[cur])
        if dist > best_dist:
            best_dist = dist
            best_phrase = cur
    return best_phrase
def eval_turney2(data_path, model_name= None, num_iterations=9999, num_context=300):

    # model = get_trained_model(model_name, path=r'D:\temp\phrase\data_art_base_loss')
    model = get_trained_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data = load_turney_data(data_path)
    # embeddings = []
    good = 0
    for i, sample_raw in enumerate(tqdm(data)): # tqdm
        sample = sample_raw[:2] + sample_raw[4:]
        cur = {}
        for phrase in sample:
            sentences = get_sentece_context(phrase)[:num_context]
            cur[phrase] = context_to_dict(sentences)
        # candidate = get_close(cur, sample[0], clac_dist_avg)
        candidate = get_far_d(cur, sample[0])
        if candidate == sample[1]:  # 0 is the gold, 1 is the label:
            good += 1
            # print(f'input: {sample}')
        # else:
        #     print(f'input: {sample}')
        #     print(f'prediction: {candidate}')
        if i % 100 == 99:
            print(f'evaluating on {i+1} samples, accurecy is {good/(i+1)}')
        if i >= num_iterations:
            return good/i
    print('result is {}'.format(good/len(data)))
def eval_turney(data_path, model_name= None, num_iterations=9999, num_context=300, model = None):

    # model = get_trained_model(model_name, path=r'D:\temp\phrase\data_art_base_loss')
    if not model:
        model = get_trained_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data = load_turney_data(data_path)
    # embeddings = []
    good = 0
    for i, sample_raw in enumerate(tqdm(data)): # tqdm
        sample = sample_raw[:2] + sample_raw[4:]
        cur = {}
        for phrase in sample:
            sentences = get_sentece_context(phrase, True)[:num_context]
            cur[phrase] = embd(sentences, phrase, tokenizer, model)
        # candidate = get_close(cur, sample[0], clac_dist_avg)

        # candidate = get_close(cur, sample[0], calc_dist_ECDF)
        candidate = get_close(cur, sample[0], calc_cosine_ECDF)
        # candidate = get_close(cur, sample[0], calc_cosine_ECDF_knn)
        if candidate == sample[1]:  # 0 is the gold, 1 is the label:
            good += 1
            # print(f'input: {sample}')
        # else:
        #     print(f'input: {sample}')
        #     print(f'prediction: {candidate}')
        if i % 100 == 99:
            print(f'evaluating on {i+1} samples, accurecy is {good/(i+1)}')
        if i >= num_iterations:
            return good/i
    print('result is {}'.format(good/len(data)))
    return good/len(data)


def get_sentence_embed(model, tokenizer, sentences):
    return torch.split(model(**tokenizer(sentences, padding=True, truncation=True, return_tensors="pt"), output_hidden_states=True,
          return_dict=True).pooler_output, 1)
def eval_turney_imp(data_path, model_name=None, num_iterations=9999, num_context=300):

    # model = get_trained_model(model_name, path=r'D:\temp\phrase\data_art_base_loss')
    model = get_trained_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data = load_turney_data(data_path)

    model = SentenceTransformer('whaleloops/phrase-bert')
    # embeddings = []
    good = 0
    # tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    # model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

    # Tokenize input texts

    # Get the embeddings
    for i, sample_raw in enumerate(tqdm(data)):
        sample = sample_raw[:2] + sample_raw[4:]
        with torch.no_grad():
            # cur = {x: torch.tensor(y) for x,y in zip(sample, get_sentence_embed(model, tokenizer, sample))}
            cur = {x: torch.tensor(y) for x,y in zip(sample, model.encode( sample, batch_size=len(sample), show_progress_bar=False))}
        candidate = get_far(cur, sample[0], clac_dot)
        # candidate = get_close(cur, sample[0], calc_dist_ECDF)
        if candidate == sample[1]:  # 0 is the gold, 1 is the label:
            good += 1
        if i % 100 == 99:
            print(f'evaluating on {i + 1} samples, accurecy is {good / (i + 1)}')
        if i >= num_iterations:
            return good / i
    print('result is {}'.format(good / len(data)))

def perform_query(query: str, dataset_name: str = "wikipediaBasic", query_type: str = "main") -> pd.DataFrame:
    template = """{{
    "queries": {{"{query_type}": "{query_content}"}},
    "data_set_name": "{dataset_name}"
    }}"""
    template = '{"queries": {"main": {"boolean": "query"}}, "data_set_name": "wikipediaBasic", "context": {"lists": {}, "tables": {}, "case_strategy": "ignore", "attempt_fuzzy": "false"}}'
    query = template.replace("query", query)
    # query = template.format(query_content=query, dataset_name=dataset_name, query_type=query_type)
    headers = {'content-type': 'application/json'}

    url, base_url = WIKIPEDIA_URL, WIKIPEDIA_BASE_URL

    response = requests.post(url, data=query.encode('utf-8'), headers=headers)
    # response = requests.post(url,
    #                          data=query.encode('utf-8'), headers=headers)

    df = get_df_from_response(response, base_url=base_url)

    return df

def plot_accu_context():
    values = [0.2733333333333333, 0.56, 0.6, 0.58, 0.6, 0.6, 0.6133333333333333, 0.6133333333333333, 0.6133333333333333,
     0.6133333333333333, 0.6133333333333333, 0.6133333333333333, 0.6133333333333333, 0.6133333333333333,
     0.6133333333333333, 0.6133333333333333, 0.6133333333333333, 0.6133333333333333, 0.62, 0.62]
    plt.ylabel('accuracy')
    plt.xlabel('# sentences in context')
    plt.plot(range(1, 1001, 50), values)
    plt.show()

def eval_list(l, plot=False):
    l = sorted(l)
    res = len(l) - sum([x[1] for x in l])
    max_res = res
    max_dist = 0
    all_res = []
    for val, label in l:
        res += label*2 - 1
        all_res.append(res /len(l))
        if res >= max_res:
            max_res = res
            max_dist = val

    if plot:
        plt.plot([x[0] for x in l], all_res)
        plt.show()
    return max_res/ len(l), max_dist

def eval_list_f1(l, plot=False):
    l = sorted(l)
    tot_pos = sum([x[1] for x in l])
    cur_pos = 0
    max_f1 = 0
    max_dist = 0
    all_res = []
    for i, (val, label) in enumerate(l):
        cur_pos += label
        p = cur_pos / i if i !=0 else 0
        r = cur_pos / tot_pos
        f1 = 2*(p*r)/(p+r) if p+r !=0 else 0
        all_res.append(f1 /len(l))
        if f1 >= max_f1:
            max_f1 = f1
            max_dist = val

    if plot:
        plt.plot([x[0] for x in l], all_res)
        plt.show()
    return max_f1/ len(l), max_dist
def compre_lemmas(p1, p2):
    d1 = context_to_dict([p1])
    d2 = context_to_dict([p2])
    for key in d1:
        if key not in d2:
            return False
        if d1[key] != d2[key]:
            return False
    return True

def get_pos(phrase):
    s = nlp(phrase)
    pos_l = [w.pos_ for w in s]
    return ' '.join(pos_l)

def get_pos_key(p1, p2):
    l = sorted([get_pos(p1), get_pos(p2)])
    return '|'.join(l)


def ppdb_eval(data_path, model_name,  num_iterations=150, num_context=150, start_index=0, model=None):
    with open(data_path) as f:
        data = json.load(f)

    threshold = 0.71
    if not model:
        model = get_trained_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    res = []

    pos = 0; neg = 0; n_d = defaultdict(list)
    for i, cur in enumerate(tqdm(data)):  #tqdm
        if i < start_index:
            continue
        p1, p2, label = cur
        # n_d[get_pos(p1)]+=1
        # n_d[get_pos(p2)]+=1
        # continue
        # if get_pos(p1) != get_pos(p2):
        #     print(f'p1: {p1}, p2:{p2}, label: {label}')
        # get_pos(p1)
        # if compre_lemmas(p1, p2) and label == 'pos':
        #     pos +=1
        #     res.append()
        # elif not compre_lemmas(p1, p2):
        #     continue
        # else:
        #     neg +=1
        # if i % 200 == 199:
        #     print(f'score is {pos/(pos+neg)}')
        # continue
        if i > num_iterations+start_index:
            break

        int_label = 1 if label == 'pos' else 0
        p1_context = get_sentece_context(p1)[:num_context]
        p2_context = get_sentece_context(p2)[:num_context]
        if len(p1_context) < 100 or len(p2_context) < 100:
            continue
        p1_embd = embd(p1_context, p1, tokenizer, model)
        p2_embd = embd(p2_context, p2, tokenizer, model)
        dist = calc_dist_ECDF(p1_embd, p2_embd)
        # dist = clac_dist_avg(p1_embd, p2_embd)
        # if (dist > threshold and label == 'neg' ) or (dist <= threshold and label == 'pos'):
        #     pass
        # else:
        #     print(f'p1: {p1}, p2: {p2}, true value: {label}, score:{dist}')
        n_d[get_pos_key(p1, p2)].append((dist, int_label))
        res.append((dist, int_label))
        if i % 200 == 199:
            print(eval_list(res, False))
            # print(n_d)
    print(n_d)
    # print(res)
    return eval_list(res, False)

def ppdb_eval_pb(data_path, num_iterations=150, start_iteration = 0):
    with open(data_path) as f:
        data = json.load(f)

    model = SentenceTransformer('whaleloops/phrase-bert')
    res = []
    for i, cur in enumerate((data)):  #tqdm
        p1, p2, label = cur

        if 'NOUN' not in get_pos(p1) and 'NOUN' not in get_pos(p2):
            continue
        int_label = 1 if label == 'pos' else 0
        p1_embd, p2_embd = model.encode([p1, p2], batch_size=2, show_progress_bar=False)
        dist = 1-clac_dot(torch.tensor(p1_embd), torch.tensor(p2_embd))
        # if (dist > threshold and label == 'neg' ) or (dist <= threshold and label):
        #     pass
        # else:
        #     print(f'p1: {p1}, p2: {p2} true value: {label}')

        res.append((dist, int_label))
        if i < start_iteration:
            continue
        if i > start_iteration + num_iterations:
            break
        if i % 200 == 199:
            print(eval_list(res, False))

    print(res)
    print(eval_list(res, False))

def bird_eval(data_path, model_name=None, num_iterations=9999, num_context=300):
    df = pd.read_csv(data_path, delimiter='\t')

    model = get_trained_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    res = []
    gold = []
    phrases = []
    contexts_len = []
    for index, row in (tqdm(df.iterrows())):
        p1 = row['term1']
        p2 = row['term2']
        p1_context = get_sentece_context(p1)[:num_context]
        p2_context = get_sentece_context(p2)[:num_context]

        p1_embd = embd(p1_context, p1, tokenizer, model)
        p2_embd = embd(p2_context, p2, tokenizer, model)
        sim = 1 - calc_dist_ECDF(p1_embd, p2_embd)
        res.append(sim)
        gold.append(row['relatedness score'])
        phrases.append(row['pair'])
        contexts_len.append((len(p1_context), len(p2_context)))
        if index % 150 == 149:
            m = max(res)
            l = sorted([(x / m, y, z, a[0], a[1]) for x, y, z, a in zip(res, gold, phrases, contexts_len)])
            res_df = pd.DataFrame(l)
            res_df.columns = ['model score', 'gold score', 'phrases', 'context1 length', 'context2 length']
            res_df.to_csv('bird.csv')
            print(f'corelation is {pearsonr(res, gold)}')
        if index > num_iterations:
            break
    print(sim)

def comapre_phrases(p1, p2, model, tokenizer, num_context, k = 1):
    p1_context = get_sentece_context(p1, data_dir=r'D:\Arts_Crafts_and_Sewing_10000/')[:num_context]
    p2_context = get_sentece_context(p2, data_dir=r'D:\Arts_Crafts_and_Sewing_10000/')[:num_context]

    p1_embd = embd(p1_context, p1, tokenizer, model)
    p2_embd = embd(p2_context, p2, tokenizer, model)
    p1_embd = p1_embd.reshape(-1, 768)
    p2_embd = p2_embd.reshape(-1, 768)
    if p1_embd.size(0) <= k or p2_embd.size(0) <= k:
        return 0
    # sim = calc_cosine_ECDF_knn(p1_embd, p2_embd, k)
    sim = calc_cosine_ECDF(p1_embd, p2_embd)
    # print(sim)
    return sim

def run_TR9856(model_name, num_context=300, num_iterations=150, k=1):
    data = pd.read_csv(r"C:\Users\Amir\web_data\similarity\IBM_Debater_(R)_TR9856.v2\IBM_Debater_(R)_TR9856.v0.2\TermRelatednessResults.csv", encoding='unicode_escape')

    model = get_trained_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    res = []
    gold = []
    phrases = []
    contexts_len = []
    for index, line in tqdm(data.iterrows()):
        p1 = line['term1']
        p2 = line['term2']
        print(f'{p1} - {p2}')
        if p1 == 'entertainment software rating':
            print(p1)
        p1_context = get_sentece_context(p1)[:num_context]
        p2_context = get_sentece_context(p2)[:num_context]

        p1_embd = embd(p1_context, p1, tokenizer, model)
        p2_embd = embd(p2_context, p2, tokenizer, model)
        p1_embd = p1_embd.reshape(-1, 768)
        p2_embd = p2_embd.reshape(-1, 768)
        if p1_embd.size(0) <= k or p2_embd.size(0) <= k:
            continue
        sim = calc_cosine_ECDF_knn(p1_embd, p2_embd, 1)
        res.append(sim)
        gold.append(line['score'])
        contexts_len.append((len(p1_context), len(p2_context)))
        if index % 150 == 149:
            # print(f'corelation is {pearsonr(res, gold)}')
            print(f'corelation is {spearmanr(res, gold)}')
        if index > num_iterations:
            break
    print(f'pearson result: {pearsonr(res, gold)}, spearman results: {spearmanr(res, gold)}')


def get_nouns_from_cluster(cluster):
    l1 = set([x[0] for x in cluster])
    l2 = set([x[1] for x in cluster])
    return list(l1.union(l2))


def embed_list(nps, model, tokenizer):
    return {np: embd(get_sentece_context(np, data_dir=r'D:\Arts_Crafts_and_Sewing_10000/')[:300], np, tokenizer, model).squeeze() for np in nps}


def cluster_eval(path, model_name, num_iterations, num_context):

    # model = get_trained_model(model_name)
    model = get_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with open(path) as f:
        data = json.load(f)
    res = []
    for i, cluster in enumerate(tqdm(data.values())):
        nps = get_nouns_from_cluster(cluster)
        res2 = []
        embd_dict = embed_list(nps, model, tokenizer)
        for j, (p1, p2, label) in enumerate(cluster):


            # print(len(get_sentece_context(p1)))
            # print(len(get_sentece_context(p2)))
            dist = calc_cosine_ECDF_knn(embd_dict[p1], embd_dict[p2])

            if label == 1:
                draw_sample(p1, p2, embd_dict[p1], embd_dict[p2], dist, label, save_location=r'temp2/pos/' + f'{p1}-{p2}.png')
            elif label == 0:
                draw_sample(p1, p2, embd_dict[p1], embd_dict[p2], dist, label, save_location=r'temp2/neg/' + f'{p1}-{p2}.png')
            res.append((dist, label))
            res2.append(dist)
        print(f'f1: {eval_list_f1(res)}')
    print(eval_list(res))
    return eval_list(res)


if __name__ == '__main__':
    cluster_eval(r"C:\Users\Amir\Downloads\initial_pairs_ori.json", 'SpanBERT/spanbert-base-cased', num_iterations=9999, num_context=300)
    exit()
    # run_TR9856('SpanBERT/spanbert-base-cased')
    # exit()
    # # print(ppdb_eval("D:\PPDB-filtered.json", 'SpanBERT/spanbert-base-cased', num_iterations=99999,start_index=11649, num_context=99999))
    # ppdb_eval_pb("D:\PPDB-filtered.json",num_iterations=150,start_iteration=11649)
    # exit()
    # bird_eval("D:\BiRD.txt", 'SpanBERT/spanbert-base-cased', num_iterations=9999, num_context=300)
    # exit()
    print(eval_turney(r"D:\turney.txt", 'SpanBERT/spanbert-base-cased', num_iterations=150, num_context=300))
    exit()
    print(eval_turney_imp(r"D:\turney.txt", 'princeton-nlp/sup-simcse-bert-base-uncased', num_iterations=9999, num_context=100))
    exit()
    # print(eval_list(raw_res, True))
    # exit()
    ppdb_eval("D:\PPDB-filtered.json", 'SpanBERT/spanbert-base-cased', num_iterations=150, num_context=300)
    exit()
    # perform_query('`binary`')
    res = []
    for i in range(1, 1001, 50):
        print(f'calculating i = {i}')
        res.append(eval_turney(r"D:\turney.txt", 'SpanBERT/spanbert-base-cased', num_iterations=150, num_context=i))
        print(res)

    # eval_turney(r"D:\turney.txt", 'SpanBERT/spanbert-base-cased')
    # eval_turney(r"D:\turney.txt", )
    # eval_turney(r"D:\turney.txt", "whaleloops/phrase-bert")