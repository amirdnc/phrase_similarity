import random
from collections import defaultdict
import torch
import numpy as np
from scipy.interpolate import interp1d
import torch.nn.functional as F
from sklearn.metrics import ndcg_score


def clac_dist_avg(l1, l2):
    l1_avg = torch.sum(l1, dim=0)/l1.size(0)
    l2_avg = torch.sum(l2, dim=0)/l2.size(0)
    return torch.cdist(l1_avg.unsqueeze(0), l2_avg.unsqueeze(0), p=2)

def clac_dist_all_vecs(l1, l2):
    dists = torch.cdist(l1, l2, p=2)
    return torch.sum(dists) / (dists.size(0)*dists.size(1))


def clac_dist_cosine_all_vecs(l1, l2):
    # dists =F.cosine_similarity(l1, l2)
    l1 = F.normalize(l1)
    l2 = F.normalize(l2)
    dists = torch.cdist(l1, l2, p=2)
    return torch.sum(dists) / (dists.size(0)*dists.size(1))

def clac_dist_median_vecs(l1, l2):
    dists = torch.cdist(l1, l2, p=2)
    res = torch.median(dists)
    # print('max: {}, min:{}, mid: {}'.format(torch.max(dists), torch.min(dists), res))
    return res
def get_interpolation(l):
    l = np.insert(l, 0, 0)
    l = np.insert(l, 0, 100)
    l_sorted = np.sort(l)
    p = 1. * np.arange(len(l)) / (len(l) - 1)
    f = interp1d(l_sorted, p)
    return f

def calc_dist_ECDF(l1, l2):
    dists = torch.cdist(l1, l2, p=2).reshape(-1).cpu().numpy()
    interpolation = get_interpolation(dists)
    return 1 - (sum(interpolation(x) for x in np.arange(0.00001, 5, 0.1))/50)

def clac_dist_alignment(l1, l2):
    dists = torch.cdist(l1, l2, p=2)
    return torch.mean(torch.min(dists, dim=1)[0])


def clac_dist_greedy_alignment(l1, l2):
    dists = torch.cdist(l1, l2, p=2)
    vals = 0
    for i in range(l1.size(0)):
        mins = torch.min(dists, dim=1)
        min_index = torch.argmin(mins[0])
        vals += dists[min_index, mins[1][min_index]]
        dists[:, mins[1][min_index]] = 9999
    return vals / l1.size(0)


def get_close(embeddings, g, dist_f):
    best_phrase = None
    best_dist = 9999
    for cur in embeddings:
        if cur == g:
            continue
        dist = dist_f(embeddings[g], embeddings[cur])
        if dist < best_dist:
            best_dist = dist
            best_phrase = cur
    return best_phrase


def cmpr(dist1, dist2):
    res = 0
    # for i in range(len(dist1)):
    #     for j in range(len(dist2)):
    sample_size = 150
    if len(dist1) * len(dist2) > sample_size:
        ii = list(range(len(dist1)))
        jj = list(range(len(dist2)))
        for i,j in zip(random.choices(ii, k=sample_size), random.choices(jj, k=sample_size)):
            if dist1[i] < dist2[j]:
                res += 1
            elif dist2[j] < dist1[i]:
                res -= 1
    else:
        for i in range(len(dist1)):
            for j in range(len(dist2)):
                if dist1[i] < dist2[j]:
                    res += 1
                elif dist2[j] < dist1[i]:
                    res -= 1
    return res



def get_close_vec(embeddings, g):
    best_phrase = None
    best_dist = None
    for cur in embeddings:
        if cur == g:
            continue
        dist = torch.cdist(embeddings[g], embeddings[cur], p=2)
        if best_dist == None or (cmpr(best_dist.reshape(-1), dist.reshape(-1)) < 0):
            best_dist = dist
            best_phrase = cur
    return best_phrase

def get_close_group(embeddings, g, dist_f, k):
    all_dists = []
    for cur in embeddings:
        if cur == g:
            continue
        dist = dist_f(embeddings[g], embeddings[cur])
        all_dists.append((dist, cur))
    all_dists = sorted(all_dists)
    return [x[1] for x in all_dists][:k]



def eval_similarity(gold, embeddings, dist):
    tot = 0
    good = 0
    for g in gold:
        if g not in embeddings:
            continue
        if not any(x in embeddings for x in gold[g]):
            continue
        tot += 1
        close_phrase = get_close(embeddings, g, dist)
        if close_phrase in gold[g]:
            good +=1
        # else:
        #     print(f'sample: {g}, prediction:{close_phrase}, golds:{gold[g]}')
    if tot == 0:
        return None
    return good/tot

def eval_ndcg(gold, sim_dict, e_context, e_words, dist1, dist2):
        we = []
        ce = []
        for g in gold:
            if g not in e_words or g not in e_context:
                continue
            if not any(x in e_words for x in gold[g]):
                continue
            if sim_dict[g.replace(' ', '_')] == 0:
                continue
            human_sim = defaultdict(int, sim_dict[g.replace(' ', '_')])
            d = get_dist_dict(e_context, e_words, g, dist1, dist2)
            reference_p = [i for i in gold if i in d and i != g]  # filter nps not in embeddings (maybe we shouldn't?)
            gold_l = [human_sim[i.replace(' ', '_')] for i in reference_p]

            we_l = [d[i][1] for i in reference_p]
            ce_l = [d[i][0] for i in reference_p]
            we.append(ndcg_score(np.asarray([gold_l]), np.asarray([we_l])))
            ce.append(ndcg_score(np.asarray([gold_l]), np.asarray([ce_l])))
        return ce, we

def eval_similarity_vec(gold, embeddings):
    tot = 0
    good = 0
    for g in gold:
        if g not in embeddings:
            continue
        if not any(x in embeddings for x in gold[g]):
            continue
        tot += 1
        close_phrase = get_close_vec(embeddings, g)
        if close_phrase in gold[g]:
            good +=1
        # else:
        #     print(f'prediction: {close_phrase}, sample: {g}, gold: {gold[g]}')
    if tot == 0:
        return None
    return good/tot

def get_close_joined(e1, e2, g, dist_f1, dist_f2, alpha):
    best_phrase = None
    best_dist = 9999
    for cur in e1:
        if cur == g:
            continue
        dist1 = dist_f1(e1[g], e1[cur])
        dist2 = dist_f2(e2[g], e2[cur])
        dist = alpha*dist1 + (1-alpha)*dist2
        dist = min(alpha* dist1, (1-alpha)*dist2)
        if dist < best_dist:
            best_dist = dist
            best_phrase = cur
    return best_phrase

def normlize(d):
    std = np.std([x[0] for x in d])
    mean = np.mean([x[0] for x in d])
    res = {x[1]: ((x[0]-mean)/std) for x in d}
    return res


def get_close_joined_adjst(e1, e2, g, dist_f1, dist_f2, alpha):
    d1 = normlize([(dist_f1(e1[g], e1[cur]), cur) for cur in e1 if cur != g])
    d2 = normlize([(dist_f2(e2[g], e2[cur]), cur) for cur in e1 if cur != g])
    d = [(max(d1[cur], d2[cur]), cur) for cur in e1 if cur != g]
    return min(d)[1]


def get_dist_dict(e1, e2, g, dist_f1, dist_f2):
    # d1 = normlize([(dist_f1(e1[g], e1[cur]), cur) for cur in e1 if cur != g])
    # d2 = normlize([(dist_f2(e2[g], e2[cur]), cur) for cur in e1 if cur != g])
    d1 = {cur: dist_f1(e1[g], e1[cur]) for cur in e1 if cur != g}
    d2 = {cur: dist_f2(e2[g], e2[cur]) for cur in e1 if cur != g}
    d = {x: (d1[x], d2[x]) for x in d2}
    return d

def get_points(gold, e_context, e_words, dist1, dist2, alpha):
    good_l = []
    bad_l = []
    names = []
    for g in gold:
        if g not in e_words or g not in e_context:
            continue
        if not any(x in e_words for x in gold[g]):
            continue
        d = get_dist_dict(e_context, e_words, g, dist1, dist2)
        t_good = []
        for good in gold[g]:# if close_phrase1 in gold[g] or close_phrase2 in gold[g]:
            if good in d:
                t_good.append((d[good], good))
                del d[good]

        if not t_good:
            continue
        good_l.append(t_good)
        bad_l.append(list(zip(d.values(), d.keys())))
        names.append(g)
    return names, good_l, bad_l

def make_res(g, p, n):
        return {'anchor': g, 'best':p[0], 'raw': p[1], 'other_best': n[0], 'dist_difference': torch.abs(p[1][0] - p[1][1])}


def get_far_points(gold, e_context, e_words, dist1, dist2, alpha):
    contexts = []
    words = []
    for g in gold:
        if g not in e_words or g not in e_context:
            continue
        if not any(x in e_words for x in gold[g]):
            continue
        d = get_dist_dict(e_context, e_words, g, dist1, dist2)

        best_context = min(d.items(), key=lambda x: x[1][0])
        best_word = min(d.items(), key=lambda x: x[1][1])
        # if best_word == best_context:
        #     continue
        if best_context[0] in gold[g] and best_word[0] in gold[g]:
            continue
        if best_context[0] in gold[g]:
             contexts.append(make_res(g, best_context, best_word))

        if best_word[0] in gold[g]:
            words.append(make_res(g, best_word, best_context))
    return contexts, words

def eval_joined_similarity(gold, e_context, e_words, dist1, dist2, alpha):
    tot = 0
    good = 0
    for g in gold:
        if g not in e_words or g not in e_context:
            continue
        if not any(x in e_words for x in gold[g]):
            continue
        tot += 1
        # close_phrase = get_close_joined(e_context, e_words, g, dist1, dist2, alpha)
        close_phrase = get_close_joined_adjst(e_context, e_words, g, dist1, dist2, alpha)
        # close_phrase1 = get_close_vec(e_context, g)
        # close_phrase1 = get_close(e_context, g, dist1)
        # close_phrase2 = get_close(e_words, g, dist2)
        if close_phrase in gold[g]:# if close_phrase1 in gold[g] or close_phrase2 in gold[g]:
            good +=1
        # else:
        #     print(f'sample: {g}, prediction:{close_phrase}, golds:{gold[g]}')
    if tot == 0:
        return None
    return good/tot

def eval_K_best_similarity(gold, e1, e2, dist1, dist2, k):
    tot = 0
    good = 0
    for g in gold:
        if g not in e2 or g not in e1:
            continue
        if not any(x in e2 for x in gold[g]):
            continue
        tot += 1
        close_phrases = get_close_group(e1, g, dist1, k)
        new_embed = {p: e2[p] for p in close_phrases if p in e2}
        new_embed[g] = e2[g]
        close_phrase = get_close(new_embed, g, dist2)
        if close_phrase in gold[g]:
            good +=1
        # else:
        #     print(f'sample: {g}, prediction:{close_phrase}, golds:{gold[g]}')
    return good/tot

