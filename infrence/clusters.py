import pickle
from collections import defaultdict
from copy import deepcopy

import spacy

from infrence.utils import get_data_with_index

my_hair = {'type': [
    'greasy hair',
    'fuller hair',
    'quality hair',
    'course hair',
    'adidas hair',
],
    'phenomena':
        ['hair loss'],

    'products': [
        'hair care',
        'hair cleaner',
        'hair shampoo',
        'hair gel',
        'hair straightener',
        'hair product',
        'hair products',
        'hair cream'],

    'treatments': [
        'hair rehab',
        'hair perm',
        'hair dye'],

    'colors': [
        'silver hair',
        'brunette hair',
        'hair color',
        'bleach hair'],

    'others': [
        'daughters hair',
        'hair growth',
        'hair stylist',
        'hair health',
        'hair l',
        'hair masque',
        'length hair']}

hair_test2 = {'1':['hair care',
'hair shampoo',
'hair gel',
'hair straightener',
'hair cleaner',
'hair product',
'hair products',
'hair dye',
'hair color',
'hair cream',
'hair masque'],
'2': ['hair loss',
'hair rehab',
'hair growth',
'hair health'],
'3':['silver hair',
'brunette hair'],
'4':['hair perm',
'bleach hair',
'0'],
'5':['quality hair',
'greasy hair',
'fuller hair',
'course hair',
'hair stylist']}

hair_test = {'type': [
    'greasy hair',
    'fuller hair',
    'quality hair',
    'course hair',
],

    'products': [
        'hair care',
        'hair cleaner',
        'hair shampoo',
        'hair gel',
        'hair straightener',
        'hair product',
        'hair products',
        'hair cream'],

    'treatments': [
        'hair rehab',
        'hair perm',
        'hair dye'],

    'colors': [
        'silver hair',
        'brunette hair',
        'hair color',
        'bleach hair']}

my_battery = {'batterys':
['car battery',
'oem battery',
'replacement battery',
'toy battery',
'trailer battery',
'auto battery',
'side battery',
'aftermarket battery',
'ups battery'
],

'adverbes':
['heavy battery',
'duty battery',
'genuine battery',
],
'operations':
['battery repair',
],

'attributes':
['battery amps',
'battery acid',
'battery connection',
],
'accesories':
['battery connector',
'battery terminal',
'battery booster',
'battery cables',
'cycle battery',
'battery cartridge',
],
'other':
['voltage battery',
'pw battery',
'battery tot',
'post battery',
'battery posts',
'apc battery',
'battery products',
'battery manufacturers'
]}

ran_battery = {'type':
['car battery',
'auto battery',
'trailer battery',
'ups battery',
'toy battery',
'side battery',
'replacement battery',
'heavy battery',
'duty battery',
'post battery',],

'accessories':
['battery connector',
'battery booster',
'battery cables',
'battery connection',
'battery cartridge',
'battery products',],

'specs':[
'battery amps'],

'brands++':[
'apc battery',
'oem battery',
'genuine battery',
'battery manufacturers',
'aftermarket battery'],

'materials & parts':[
'battery acid',
'battery posts',
'battery terminal'],

'not clear':[
'pw battery',
'battery tot',
'cycle battery',
'battery repair',
'voltage battery']}

hila_phone = {'extras': [
    'phone cover',

    'phone case',


    'phone charger',

    'phone holder',

    'phone holster',
    'phone cases']

    , 'parts': [

        'phone screen',

        'phone battery'],

    'properties(?)': [

        'phone charging',

        'phone usage',

        'phone look',

        'phone life'],

    'brands': [

        'samsung phone',

        's2 phone',

        'phone 4s',

        'g phone',

        'galaxy phone',

        'kitty phone']

    , 'others': [

        'daughters phone',

        'sons phone',

        'cell phone',

        'phone model',

        'phone user',

        'touch phone',

        'phone stores'],

    'Garbage': [

        'phone fine',

        'phone shows',

        'phone okay',

        'phone unbroken'],

    'Couldn’t figure out':

        ['phone pretting']}

hila_battery = {'types':[
'toy battery',
'replacement battery',
'genuine battery',
'ups battery',
'oem battery',
'side battery',
'cycle battery',
'duty battery',
'heavy battery'],
'brands':[
'aftermarket battery',
'apc battery',
'pw battery'],

'add-ons':[
'battery connector',
'battery cartridge',
'battery cables',
'battery booster',
'battery products'],

'parts':[
'battery terminal',
'battery connection'],

'no-name':[
'trailer battery',
'auto battery',
'car battery'],

'others':[
'battery manufacturers',
'battery repair',
'battery amps',
'battery acid'],

'don’t know what this is':[
'battery tot',
'voltage battery',
'post battery',
'battery posts']}

ran_paper = {'printer paper types':[
'parchment paper',
'release paper',
'block paper',
'copy paper',
'quality paper'],

'kids paper types':[
'construction paper',
'kids paper',
'scrap paper'],

'art paper types':[
'vellum paper',
'translucent paper',
'tracing paper',
'watercolor paper'],

'other paper types':[
'paper transfer',
'transfer paper',
'photo paper',
'graph paper',
'paper wrapper',
'paper backing'],

'tools for papers':[
'paper knife',
'paper piecer',
'paper piece',
'paper piecing'],

'made of paper':[
'paper towel'],

'misc-1':[
'paper insert',
'insert paper'],

'misc-2':[
'paper classes'],

'misc-3':[
'paper setting'],

'misc-4':[
'paper instructions'],

'misc-5':[
'foundations paper'],

'brands':[
'arches paper']}


my_paper = {
'technical paper':['graph paper',
'foundations paper',
'translucent paper',
'vellum paper',
'copy paper',
'tracing paper',
'watercolor paper',
'paper towel',
'paper wrapper',
'scrap paper',
'block paper',
'kids paper',
'paper backing',
'transfer paper',
'photo paper',
'construction paper'
],
'parts of paper':['paper piecing',
'paper piece',
],
'tools':['paper piecer',
'paper knife',
'paper insert',
'insert paper'
],
'other':['paper classes',
'quality paper',
'paper transfer',
'arches paper',
'paper setting',
'paper instructions',
'parchment paper',
'release paper',
]
}

# {
#     'types':['baby_powder_smell''banana_smell', 'clean_fresh_smell', 'earthy_smell'],
#      'clean_smell_conditions', 'cologne_type_smell', 'delicious_smell', 'entire_hotel_room_smell', 'fresh_smell', 'hair_smell', 'hotel_room_smell', 'light_clean_smell', 'light_smell', 'nature_smell', 'oily_smell', 'perfume_smell', 'smell_conditions', 'smell_of_fresh_grapefruit', 'smell_of_lemons', 'vague_smell']}
#
# }
def cluster_to_list(c):
    indexes = []
    for i, sublist in enumerate(c.values()):
        indexes.extend([i]*len(sublist))
    return [item for sublist in c.values() for item in sublist], indexes


def get_similar_words(w, l):
    for words in l.values():
        if w in words:
            return words
    return []

def gen_from_clusters():
    clusters = [hair_test, hair_test2]
    is_test = True
    l = cluster_to_list(clusters[0])[0]
    # for
    word = []
    others = []
    golds = []
    for i in range(len(l)):
        if l[-1] == '0':
            continue
        word.append(l.pop())
        # others.append(deepcopy(l))
        others.append(','.join(l).replace(' ', '_'))
        # others.append(l)

        l.insert(0,word[-1])
        gold = []
        for c in clusters:
            gold.extend(get_similar_words(word[-1], c))
        gold = list(set(gold))
        gold = [x.replace(' ', '_') for x in gold if x != word[-1]]
        golds.append(','.join(gold).replace(' ', '_'))
        # golds.append(gold)

    df = pd.DataFrame.from_dict({'word':word, 'similars':others, })
    if is_test:
        df['choose_gold'] = golds
        df['_golden'] = 'TRUE'
    else:
        df['_golden'] = 'FALSE'
    df['choose_gold_reason'] = ''

    df = df.iloc[:10]
    df.to_csv('out.tsv', sep='\t', index=False)

def collect_by_noun(phrases):
    d = defaultdict(list)
    for phrase, freq in phrases:
        for p in phrase.split(' '):
            d[p].append((freq, phrase))
    singles = defaultdict(int)
    for k in list(d.keys()):
        d[k] = list(set(d[k]))
        if len(d[k]) == 1:
            singles[d[k][0]] += 1
            del d[k]
    singles_l = [k for k,v in singles.items() if v >= 2]
    return d, singles_l



def top_nouns(nouns):
    return

nlp = spacy.load("en_core_web_sm")
def get_dep(doc):
    for t in doc:
        if t.dep_ == 'ROOT' or t.dep_ == 'prep' or t.dep_ == 'det':
            continue
        return t.dep_

def filter_nouns(nouns_l, limit):
    res = []
    lemmas = set()
    for n in nouns_l:
        doc = nlp(n)
        lemma = ' '.join(x.lemma_ for x in doc)
        if lemma in lemmas:
            # print('lemma: {}, sentence: {}'.format(lemma, doc))
            continue
        lemmas.add(lemma)
        if get_dep(doc) != 'amod':
            # print([t.dep_ for t in doc])
            # print(doc)
            res.append(n)
            if len(res) >= limit:
                return res

bad_l = ['black', 'end', 'original', 'rear', 'front', 'lights', 'of','my', 'type', 'white', 'box', 'old', 'piece', 'product', 'projects', 'different', 'a', 'nice', 'set', 'yarn', 'the', 'many', 'small', 'own', 'new','this', 'diffrent', 'little', 'lot', 'many', 'size', 'large', 'first', 'several', 'few', 'big', 'beautiful', 'case', 'cases', 'protector', 'cover', 'samsung', 'galaxy' ]
def gen_from_data(path):
    num_nouns = 15
    num_words = 20
    index, data = get_data_with_index(path, 2000000)

    # d = find_compounds(data[:10000])
    # d = find_compounds(data)
    pickle_path = r'arts_dict.pickle'
    pickle_path = r'Cell_Phones.pickle'
    pickle_path = r'Automotive_5'
    # with open(pickle_path, 'wb') as handle:
    #     pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(pickle_path, 'rb') as handle:
        d = pickle.load(handle)
    group, singles = collect_by_noun(d.items())
    noun_list = sorted([(len(v), k) for k, v in group.items() if k not in bad_l], reverse=True)
    word = []
    others = []

    print('Nouns: {}'.format(noun_list[:num_nouns + 10]))
    for f, noun in noun_list[:num_nouns]:
        l = filter_nouns([x[1] for x in sorted(group[noun], reverse=True)], num_words)
        # l = [x[1] for x in sorted(group[noun], reverse=True)][:num_words]
        print('{}: {}'.format(noun, l))
        for i in range(len(l)):
            word.append(l.pop())
            others.append(','.join(l).replace(' ', '_'))
            l.insert(0, word[-1])

    df = pd.DataFrame.from_dict({'word':word, 'similars':others, })
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv('out.tsv', sep='\t', index=False)



if __name__ == '__main__':
    import pandas as pd
    # gen_from_clusters()
    path = r'All_Beauty_5.json.gz'
    path = r'Arts_Crafts_and_Sewing_5.json.gz'
    path = r'Cell_Phones_and_Accessories_5.json.gz'
    path = r'Automotive_5.json.gz'
    gen_from_data(path)



