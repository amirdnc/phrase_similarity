import itertools
import json
from collections import defaultdict
# import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import itertools

def get_common_word(x):
    d = defaultdict(int)
    for phrase in x['similars'].split('\n')[:5]:
        for w in phrase.split('_'):
            d[w] += 1
    word = sorted(d.items(), key=lambda x:x[1], reverse=True)[0][0]
    return word

def get_edge(x):
    return tuple(sorted([x['word'], x['choose']]))

def gen_clusters_from_similarity(path):
    data = pd.read_csv(path)
    data['common'] = data.apply(get_common_word, axis=1)
    # print(data['common'])
    for noun in set(data['common'].to_list()):
        edges_d = defaultdict(int)
        edges = data[noun == data['common']].apply(get_edge, axis=1).to_list()
        print(edges)
        for e in edges:
            edges_d[e] +=1
        nodes = set([item for sublist in edges for item in sublist])
        G = nx.Graph()
        G.add_nodes_from(nodes)
        for e, w in edges_d.items():
            if w > 2:
                G.add_edge(e[0], e[1], weight=w)
        if '0' in nodes:
            G.remove_node('0')
        if '1' in nodes:
            G.remove_node('1')

        nx.draw_networkx(G,with_labels=True)

        # plt.savefig("simple_path.png") # save as png
        plt.show() # display


def process_jason(path):
    clusters = []
    with open(path) as f:
        for l in f:
            t = json.loads(l)
            for sim in t['results']['judgments']:
                if '1' in sim['data']['choose']:
                    continue
                if '0' in sim['data']['choose']:
                    clusters.append(t['data']['word'])
                    continue
                sim['data']['choose'].extend(t['data']['word'])
                clusters.append(sim['data']['choose'])
    return clusters

def gen_similarities(path):
    data = process_jason(path)
    couples = defaultdict(int)
    singles = defaultdict(int)
    for d in data:
        for l in itertools.combinations(d, 2):
            l = sorted(l)
            # name = '{} {}'.format(l[0], l[1])
            couples[name] += 1
        for w in d:
            singles[w] += 1

    connections = defaultdict(list)
    for k, v in singles.items():
        for couple in couples:
            if k in couple:
                phrase = couple.replace(k, ' ').strip()
                if phrase.startswith('_') or phrase =='ing':
                    continue
                connections[k].append((phrase, couples[couple]/v))
    for k,v in connections.items():
        mod_v = [x for x in v if x[1] > 0.3]
        mod_v = sorted(v, reverse=True, key=lambda x: x[1])
        print(f'{k}: {mod_v}')
    return connections
    # print(couples)
    # print(singles)




if __name__ == '__main__':
    # gen_clusters_from_similarity(r"D:\human\f1811941.csv")
    gen_similarities(r"D:\human\c_art.json")
    # gen_similarities(r"job_1854561.json")

