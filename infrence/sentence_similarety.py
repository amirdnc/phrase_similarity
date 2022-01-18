import gzip
import random
import time
from collections import defaultdict
from math import ceil

import nltk
from scipy.sparse import coo_matrix, hstack, csr_matrix, vstack
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.manifold import TSNE
from sklearn.metrics import homogeneity_score, v_measure_score
# from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import zipfile
from nltk.tokenize import sent_tokenize

# Mean Pooling - Take attention mask into account for correct averaging
from infrence.imports import get_trained_model
from infrence.clusters import my_battery, ran_battery, hila_battery, my_paper, ran_paper, my_hair

from infrence.utils import get_data_with_index


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# Sentences we want sentence embeddings for
sentences = ['This framework generates embeddings for each input sentence',
             'Sentences are passed as a list of string.',
             'The quick brown fox jumps over the lazy dog.']
sentences = ['body wash', 'shower gel', 'de provence', 'pre de', 'smells great', 'love shampoo', 'leaves skin',
             'shampoo conditioner', 'smells good', 'love scent', 'makes hair', 'leaves hair', 'dry skin', 'eye cream',
             'using product', 'hard find', 'body lotion', 'love smell', 'highly recommend', 'shea butter',
             'great shampoo', 'smells like', 'hair feels', 'lathers well', 'skin feels', 'best shampoo', 'skin soft',
             'bath body', 'sensitive skin', 'hair feel', 'smells amazing', 'get compliments', 'fine hair', 'bar soap',
             'hair soft', 'body cream', 'bubble bath', 've tried', 'love stuff', 'ever used', 'calibra eye',
             'goes long', 'started using', 'ca find', 'skin feel', 'love body', 'wash hair', 'favorite body',
             'skin feeling', 'curly hair', 'could find', 'recommend product', 'using years', 'smell great',
             'excellent product', 'old spice', 'high quality', 'really good', 'dry hair', 'love fragrance',
             'bath shower', 'hair product', 'cleans hair', 'works well', 'makes skin', 'thin hair', 'feel clean',
             'hair looks', 'feeling clean', 'provence soaps', 'product love', 'green tea', 'hair great', 'top coat',
             'use conditioner', 'favorite fragrance', 'lavender blossoms', 'body washes', 'find stores', 'love shower',
             'pro health', 'last long', 'long way', 'coarse hair', 'skin moisturized', 'favorite scent', 'clean soft',
             'flat iron', 'leave hair', 'fresh scent', 've using', 'hair love', 'get rid', 'great smell',
             'product using', 'hair much', 'hair falling', 'made hair', 'hair feeling', 'hair ca', 'wish could',
             'smell like', 'used product', 'would love', 'smells wonderful', 'lectric shave', 'using shampoo',
             'thick hair', 'happy find', 'product really', 'looks like', 'alcohol free', 'body works', 'smell nice',
             'silky smooth', 'clean fresh', 'little goes', 'long lasting', 'used years', 'feels like', 'glad found',
             'fine lines', 'use product', 'smell good', 'great hair', 'husband loves', 'product arrived', 'feels clean',
             'crest pro', 'hair ve', 'product years', 'fragrance love', 'come back', 'lasts long', 'stopped making',
             'smooth soft', 'clean smell', 'smells nice', 'fresh smell', 'beauty supply', 'subtle scent', 'around eyes',
             'ca get', 'love way', 'give try', 'also great', 'could get', 'love wish', 'skin care', 'bare minerals',
             'great job', 'able buy', 'love makes', 'good stuff', 'best stuff', 'feels great', 'stuff smells',
             'shaving cream', 'longer available', 'smell fresh', 'keep using', 'fresh clean', 'find amazon',
             'every time', 'bring back', 'buy love', 'product use', 'good deal', 'nice scent', 'let dry', 'nail polish',
             'find love', 'feeling like', 'stores anymore', 'time love', 'received product', 'two bottles',
             'great stuff', 'products good', 'find anywhere', 'price love', 'use every', 'product ve',
             'product excellent', 'time tried', 'product hard', 'product fine', 'pricey worth', 'perfect size',
             'wonderful product', 'every penny', 'worth every', 'years love', 've found', 'soft smooth', 'high end',
             'like product', 'get good', 'price great', 'able find']
# sentences = ['works well', '3d printer', 'duct tape', 'work great', 'ie utf8', 'high quality', '3d printing', 'product link', 'link linked', 'hook product', 'data hook', 'normal href', 'ref cm_cr_arp_d_rvw_txt', 'link normal', 'class link', 'linked class', 'cm_cr_arp_d_rvw_txt ie', 'heavy duty', 'highly recommend', 'nbsp data', 'good stuff', 'prints well', 'works good', 'hot end', 'heat shrink', 'power supply', 'easy read', 'print bed', 'heated bed', 'amazon com', 'great quality', 'excellent product', 'great stuff', 'pretty good', 'super glue', 'wd 40', 'stainless steel', 'around house', 'input type', 'video block', 'hidden name', 'class video', 'type hidden', 'quality good', 'images amazon', 'name value', 'com images', 'ssl images', 'na ssl', 'value https', 'images na', 'https images', 'every time', 'quality product', '3d printers', 'build plate', 'looks like', 'work fine', 'price great', 'shop vac', 'great filament', 'last long', 'print quality', 'electrical tape', 'pla filament', 'works expected', 'great value', 'exactly described', 'sd card', 'prints great', 'really good', 'gorilla tape', 've tried', 'works advertised', 've using', '3d solutech', 'worked perfectly', 'seems work', 'easy install', 'works like', 'ever used', 'gorilla glue', 'hatchbox filament', 'works perfectly', 'good job', 'well worth', 'could find', 'stuff works', 'seems like', 'worked fine', 'good deal', 'well built', 'job done', 'first aid', 'well packaged', 'great job', 'pla abs', 'filament prints', 'table saw', 'work good', 'works described', 'could get', 'great item', 'far good', 'quality great', 'shrink tubing', 'prusa i3', 've found', 'big deal', 'filament ve', 'excellent quality', 'quality prints', 'great works', 'digital caliper', 'on off', 'go wrong', 'seems accurate', 'drill press', 'much say', 'nice quality', 'exactly advertised', 'seems good', 'different sizes', 'first layer', 'layer adhesion', 'drill bit', 'put together', 'quality price', 'good enough', 'pretty well', 'exactly needed', 'creator pro', 'jb weld', 'described works', 'ca beat', 'print head', 'painters tape', 'glue stick', 'fair price', 'hold well', 'get good', 'sticks well', 'would use', 'flashforge creator', 'made usa', 'vacuum cleaner', 'fast delivery', 'hardware store', 'brush head', 'perfect size', 'price point', 'worth money', 'spacing top', 'quality filament', 'also used', 'price works', 'working well', 'spool holder', 'slate img', 'video slate', 'spacing small', 'mini video', 'class section', 'img url', 'video url', 'small spacing', 'block div', 'section spacing', 'div input', 'top mini', 'div id', 'url input', 'id video', 'png class', 'easy clean', 'mp4 class', 'able get', 'right size', 'great tape', 'time tell', 'used yet', 'get pay', 'brush heads', 'glass bed', 've printed', 'well used', 'great tool', 've never', 'tape works', 'tool box', 'better quality', 'price right', 'less expensive', 'well great', 'working great', 'recommend product', 'easy work', 'url nbsp', 'good buy', 'easier use']
sentences = ['fragrance', 'adorable compact fragrance', 'nice fragrance', 'great fragrance', 'time favorite fragrance',
             "70 's fragrance", 'fragrance gods', 'wonderful fragrance', 'diva fragrance', 'signature fragrance',
             'good fragrance', 'classy , elegant fragrance', 'awesome fragrance', 'nice fruity fragrance',
             'nice mild fragrance', 'favorite fragrance', 'lovely powder bomb fragrance', 'particular fragrance',
             'heavy fragrance', 'sexy fragrance', 'shower gel fragrance', 'amazing fragrance', 'citrus fragrance',
             'bergamot fragrance', 'soft femine fragrance', 'far favorite fragrance', 'beautiful fragrance',
             'every single escada fragrance', 'fragrance free option', 'unique lovely fragrance', 'smell',
             'vague smell', 'great smell', 'banana smell', 'oily smell', 'delicious smell', 'light clean smell',
             'light smell', 'fresh smell', 'clean smell conditions', 'earthy smell', 'best smell', 'cologne type smell',
             'wonderful smell', 'nice smell', 'amazing smell', 'foodie smell', 'masculine smell', 'clean & fresh smell',
             'subtle spice smell', 'pleasant smell', 'simple clean smell', 'faint smell', 'light , breezy smell',
             'divine smell', 'lavender smell', "'' smell", 'strong smell', 'clean smell', 'lovely smell']
# sentences = ['power', 'power hand tools', 'suction power', 'power tools', 'new vacuum power tool', 'power company', 'holding power', 'lasting power', 'power tool', 'good holding power', 'battery power duraspin', 'enough power', 'nailer power', 'good suction power', 'power cord', 'power button', 'orange power cord', 'great lasting power', 'seemingly endless steam power', 'power cords', 'great holding power', 'outlet supplying power', 'almost power cord', 'power outlet', 'worn power outlets', 'foreign power cord devices', 'power wire', 'high low power devices', 'low power devices', 'much power']
# sentences = ['battery', 'car', 'power', 'quality', 'oil', 'cable', 'replacement', 'engine', 'motor', 'type', 'repair', 'apc', 'product', 'filter', 'wheel', 'truck', 'system', 'pump', 'pad', 'factory', 'tool', 'shop', 'price', 'air', 'fan', 'duty', 'wax', 'time', 'plug', 'lock', 'head', 'gauge', 'dust', 'amp', 'wire', 'rubber', 'polisher', 'plastic', 'model', 'inverter', 'box', 'auto', 'ac', 'paper', 'line', 'home', 'charger', 'cap', 'work', 'wool', 'use', 'side', 'mower', 'metal', 'fluid', 'vehicle', 'unit', 'service', 'generator', 'gas', 'drain', 'buffer', 'tire', 'part', 'maintenance', 'light', 'fuel', 'equipment', 'end', 'duster', 'vac', 'trailer', 'style', 'size', 'oem', 'material', 'lite', 'kit', 'job', 'heat', 'handle', 'design', 'cover', 'copper', 'computer', 'area', 'amazon', 'wash', 'voltage', 'torque', 'store', 'plate', 'piece', 'parking', 'output', 'inch', 'hp', 'gasket', 'floor', 'charge', 'bag', 'wood', 'wiring', 'wave', 'velcro', 'valve', 'tube', 'test', 'surface', 'spout', 'shipping', 'set', 'radio', 'pickup', 'neck', 'machine', 'jeep', 'honda', 'hand', 'garage', 'frame', 'foam', 'fisher', 'etc', 'cycle', 'cleaner', 'case', 'cartridge', 'brand', 'book', 'body', 'b', 'agm', 'water', 'warranty', 'vacuum', 'storage', 'star', 'speed', 'source', 'shaft', 'section', 'screw', 's', 'rv', 'road', 'problem', 'pan', 'owner', 'mop', 'mechanic', 'loop', 'length', 'item', 'information', 'hook', 'hole', 'guide', 'flare', 'fit', 'emergency', 'diesel', 'clamp', 'civic', 'buffing', 'brush', 'bone', 'blade', 'backup', 'attachment', 'window', 'version', 'tree', 'switch', 'supply', 'stick', 'step', 'space', 'seat', 'sander', 'rpm', 'rivet', 'ring', 'recommend', 'range', 'project', 'pressure', 'period', 'paint', 'pack', 'number', 'mini', 'marine', 'man', 'love', 'lot', 'lomanco', 'house', 'hose', 'guy']  # automotive
sentences = ['tape', 'quality', 'metal', 'glue', 'wood', 'size', 'duty', 'shop', 'tool', 'plastic', 'gun', 'work',
             'vac', 'product', 'hose', 'brush', 'brand', 'box', 'hole', 'door', 'home', 'water', 'head', 'screw',
             'dust', 'air', 'wire', 'vacuum', 'paper', 'dewalt', 'nailer', 'kit', 'drill', 'area', 'wheel', 'use',
             'type', 'system', 'switch', 'm', 'inch', 'grommet', 'window', 'steel', 'sander', 'power', 'cleaning',
             'bit', 'attachment', 'store', 'hand', 'wall', 'stuff', 'safety', 'kitchen', 'floor', 'filter', 'end',
             'tank', 'table', 'storage', 'step', 'spray', 'shipping', 'repair', 'project', 'machine', 'kreg', 'item',
             'grinder', 'container', 'cement', 'cable', 'sheet', 'set', 'scotch', 'saw', 'room', 'pipe', 'nail', 'job',
             'indicator', 'hoover', 'hammer', 'dispenser', 'company', 'cleaner', 'case', 'carpet', 'bag', 'amazon',
             'unit', 'sound', 'side', 'rubber', 'price', 'plywood', 'gorilla', 'foam', 'driver', 'collection', 'caulk',
             'base', 'pack', 'material', 'jointer', 'irwin', 'iron', 'grade', 'gallon', 'diameter', 'cabinet', 'block',
             'weather', 'trash', 'strength', 'stove', 'spring', 'silicone', 'senco', 'replacement', 'pressure', 'port',
             'point', 'piece', 'part', 'man', 'love', 'lock', 'load', 'ii', 'hardware', 'grizzly', 'grip', 'gas',
             'fuel', 'equipment', 'drywall', 'cup', 'bottle', 'arrow', 'washer', 'version', 'unibit', 'truck', 'tile',
             'temperature', 'supply', 'super', 'steam', 'star', 'self', 'sealant', 'seal', 'review', 'rating',
             'profile', 'precision', 'pocket', 'packaging', 'outlet', 'neck', 'name', 'max', 'light', 'import', 'ie',
             'hinge', 'heat', 'handle', 'guard', 'garbage', 'garage', 'furniture', 'freight', 'frame', 'flex', 'finish',
             'fine', 'festool', 'design', 'description', 'cover', 'control', 'color', 'cm_cr_arp_d_rvw_txt', 'clamp',
             'car', 'capacity', 'cap', 'button', 'brute', 'belt', 'assortment', 'anvil', 'aluminum', 'weight', 'velcro',
             'usa', 'tube']  # science industry
sentences = ['hair', 'water', 'shampoo', 'conditioner', 'wash', 'body', 'fragrance', 'cream', 'product', 'bath', 'soap',
             'scent', 'lotion', 'health', 'skin', 'vanilla', 'oil', 'gel', 'crest', 'line', 'lavender', 'tea', 'spray',
             'spice', 'powder', 'gift', 'care', 'smell', 'shower', 'price', 'wintergreen', 'size', 'rinse', 'perfume',
             'moisture', 'eye', 'chemical', 'bottle', 'amazon', 'vitamin', 'unit', 'type', 'travel', 'time', 'thing',
             'silver', 'sea', 'quality', 'love', 'hibiscus', 'hbl', 'greasy', 'formula', 'extract', 'cologne', 'vera',
             'tree', 'taste', 'supply', 'shine', 'shave', 'shampoos', 'set', 'ross', 'pure', 'pattern', 'mouthwash',
             'mouth', 'morning', 'man', 'lip', 'lauder', 'julep', "j'adore", 'growth', 'face', 'end', 'dose', 'day',
             'cookie', 'color', 'citrus', 'christmas', 'case', 'butter', 'brush', 'bee', 'beauty', 'bar', 'baby', 'b',
             'almond', 'allergy', 'zum', 'xl', 'waterpik', 'velva', 'vegan', 'use', 'treatment', 'trade', 'tongue',
             'tiare', 'thermique', 'texture', 'system', 'sweet', 'summer', 'sugar', 'straightener', 'store', 'star',
             'spring', 'speed', 'silky', 'side', 'shop', 'shadow', 'service', 'sensation', 'search', 'scalp',
             'savannah', 'room', 'riche', 'review', 'residue', 'replacement', 'rental', 'queen', 'q10', 'professionnel',
             'power', 'point', 'photoshop', 'photo', 'perm', 'parma', 'oz', 'orchid', 'norelco', 'multiprotection',
             'model', 'mint', 'milk', 'matrix', 'masque', 'makeup', 'luxury', 'life', 'le', 'kelp', 'job', 'itchy',
             'ingredient', 'home', 'herbal', 'herbaflor', 'hand', 'gum', 'goddess', 'gloss', 'fruity', 'francisco',
             'foxcrest', 'foundation', 'food', 'flower', 'filler', 'favorite', 'fall', 'escada', 'endurance', 'e',
             'dial', 'design', 'delivery', 'd', 'customer', 'creamy', 'condition', 'commission', 'collection',
             'coconut', 'cleaning', 'cleaner', 'claw', 'choice', 'chilitis', 'chanel', 'candy', 'brown', 'box', 'blue',
             'bleach', 'biolage', 'berry', 'bean', 'battery', 'base']  # all buety
sentences = ['greasy hair', 'hair care', 'hair cleaner', 'hair shampoo', 'hair loss', 'daughters hair', 'hair rehab',
             'silver hair', 'hair growth', 'hair gel', 'hair straightener', 'hair product', 'fuller hair',
             'quality hair', 'hair products', 'hair stylist', 'hair perm', 'hair health', 'hair l', 'brunette hair',
             'hair color', 'hair dye', 'hair cream', 'hair masque', 'length hair', 'bleach hair', 'course hair',
             'adidas hair']
sentences = sentences[:50]
# Load AutoModel from huggingface model repository

if __name__ =='__main__':
    #sbert
    model_name = 'SpanBERT/spanbert-base-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).cuda()

    #triplets
    # path = r"C:\Users\Amir\Dropbox\workspace_python\BERT-Relation-Extraction\data\task_test_checkpoint_3.pth.tar"
    model = get_trained_model('SpanBERT/spanbert-base-cased')
    # checkpoint = torch.load(path)
    # model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mask_id = tokenizer.convert_tokens_to_ids('[MASK]')

def hierchy(sentence_embeddings):
    linked = linkage(sentence_embeddings, 'average')
    plt.figure(figsize=(40, 21))
    dendrogram(linked,
               orientation='top',
               labels=sentences,
               distance_sort='descending',
               show_leaf_counts=True)
    plt.show()


def tsne(sentence_embeddings):
    X = np.array(sentence_embeddings)
    X_embd = TSNE(n_components=2).fit_transform(X)
    # print(X_embd)
    x = X_embd[:, 0]
    y = X_embd[:, 1]
    plt.scatter(x, y)
    for i, text in enumerate(sentences):
        plt.annotate(text, (x[i], y[i]))
    plt.show()


def sbert():
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # hierchy
    linked = linkage(sentence_embeddings, 'single')
    plt.figure(figsize=(40, 21))
    dendrogram(linked,
               orientation='top',
               labels=sentences,
               distance_sort='descending',
               show_leaf_counts=True)
    plt.show()


def get_word_embding(sentences):
    path = r"D:\glove\glove.6B.200d.txt"
    embeddings_dict = {}

    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    arrs = []
    for s in sentences:
        ss = s.split(' ')
        count = 0
        arr = np.array([0] * 200, dtype='float64')

        for str in ss:
            if str == 'power' or str == 'fragrance' or str == 'hair':
                continue
            if str in embeddings_dict:
                count += 1
                arr += embeddings_dict[str]
            else:
                print('no string {}'.format(str))
        if count != 0:
            arrs.append(arr / count)
        else:
            arrs.append(arr)
    return arrs


def get_indexes(w1, w2, indexes):
    if w1 in indexes and w2 in indexes:
        return set.intersection(set(indexes[w1]), set(indexes[w2]))
    else:
        return set()

def get_indexes_imp(ws, indexes):
    if ws[0] in indexes:
        cur = set(indexes[ws[0]])
    for w in ws:
        if w not in indexes:
            return set()
        cur = set.intersection(cur, set(indexes[w]))
    return cur
voc = {}
voc_indx = 0

def get_voc_indx(w):
    global voc_indx
    if w in voc:
        return voc[w]
    else:
        voc[w] = voc_indx
        voc_indx += 1
        return  voc[w]


def dense_embd(final_sents):
    batch_size = 128
    sents = []
    for i in range(ceil(len(final_sents) / batch_size)):
        cur_sent = final_sents[batch_size * i:batch_size * (i + 1)]
        encoded_input = tokenizer(cur_sent, padding=True, truncation=True, max_length=128, return_tensors='pt').to(
            'cuda')
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
            # model_output = model(encoded_input['input_ids'])
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).cpu().detach()
        sents.append(sentence_embeddings)
    total_embd = torch.cat(sents, dim=0)
    return (total_embd.sum(0) / len(final_sents)).numpy()

def padded_tensor(arr):
    b = np.zeros([len(arr), len(max(arr, key=lambda x: len(x)))])
    for i, j in enumerate(arr):
        b[i][0:len(j)] = j
    return torch.tensor(b,dtype=torch.long).cuda()

def castum_embd(final_sents):
    batch_size = 128
    sents = []
    for i in range(ceil(len(final_sents) / batch_size)):
        cur_sent = final_sents[batch_size * i:batch_size * (i + 1)]
        encoded_input = tokenizer(cur_sent)
        # Compute token embeddings
        index = []
        for sen in encoded_input['input_ids']:
            index.append([i for i,x in enumerate(sen) if x == mask_id])
        index = torch.tensor(index).cuda()
        # encoded_input = encoded_input.to('cuda')
        with torch.no_grad():
            model_output = model(padded_tensor(encoded_input['input_ids']), torch.tensor(index).squeeze().to('cuda'))

        sents.append(model_output)
    total_embd = torch.cat(sents, dim=0)
    return (total_embd.sum(0).cpu() / len(final_sents)).numpy()

vec_size = 20000
def tf_idf_embd(final_sents):
    v = defaultdict(int)
    for s in final_sents:
        for w in nltk.word_tokenize(s):
            v[get_voc_indx(w)] += 1
    indx = []
    val = []
    for i in v:
        indx.append(i)
        val.append(v[i])
    mat = csr_matrix((val, (np.zeros(len(indx)), indx)), shape=(1, vec_size))
    return mat

def get_sentence_embedding(sentences, data_filename):
    index, data = get_data_with_index(data_filename, number=200000)
    res = []
    for phrase in sentences:
        # phrase = 'pw battery'
        words = nltk.word_tokenize(phrase)
        indx = get_indexes(words[0], words[1], index)
        print(phrase)
        final_sents = []
        for i in indx: #  Shold this be comented out?
            for s in sent_tokenize(data[i]):
                if s in final_sents:
                    continue
                if any(w not in s for w in words):
                    continue
                final_sents.append(s)
        final_sents = [x.replace(phrase, '[MASK]') for x in final_sents if phrase in x]
        if not final_sents:
            # res.append(np.zeros(768))
            res.append(np.zeros(28996))
            continue
        # embd = dense_embd(final_sents)
        embd = castum_embd(final_sents)

        # embd = tf_idf_embd(final_sents)
        res.append(embd)
    # res = vstack(res)
    res = np.asarray(res)
    # res = np.nan_to_num(res / (res != 0).sum(axis=0))
    return res



def cluster_to_list(c):
    indexes = []
    for i, sublist in enumerate(c.values()):
        indexes.extend([i]*len(sublist))
    return [item for sublist in c.values() for item in sublist], indexes


def print_clusters(sentences, indexes):
    d= defaultdict(list)
    for name, i in zip(sentences, indexes):
        d[i].append(name)

    for i in d:
        print('Cluster {}:'.format(i))
        for v in d[i]:
            print('\t{}'.format(v))


def validate_clusters(s1, s2, i2):
    d = {s: i for s,i in zip(s2, i2)}
    res = [d[s] for s in s1]
    return res


def run_rand_baseline():

    res = []
    indexes = [random.randint(0, 5) for i in range(30)]
    for t in range(30):
        preds = []
        for i in range(len(indexes)):
            preds.append(random.randint(0, 5))
        res.append(v_measure_score(indexes, preds))
        # res.append(homogeneity_score(indexes, preds))
    print('mean: {}, std: {}'.format(np.mean(res), np.std(res)))
    exit()
def eval_clusters_sim():
    s1, i1 = cluster_to_list(hila_battery)
    s2, i2 = cluster_to_list(ran_battery)
    i2 = validate_clusters(s1, s2, i2)
    print(v_measure_score(i1, i2))
    exit()
if __name__ == '__main__':
    # eval_clusters_sim()
    # run_rand_baseline()
    # sentences, indexes = cluster_to_list(clusters)
    # sentences, indexes = cluster_to_list(my_battery)
    # sentences, indexes = cluster_to_list(my_paper)
    sentences, indexes = cluster_to_list(my_hair)


    # arrs = get_word_embding(sentences)
    # arrs = get_sentence_embedding('paper', sentences, 'Industrial_and_Scientific_5.json.gz')
    # arrs = get_sentence_embedding('battery', sentences, 'Automotive_5.json.gz')
    arrs = get_sentence_embedding(sentences, 'All_Beauty_5.json.gz')

    est = KMeans(n_clusters=6)
    # est = AffinityPropagation(random_state=6)
    est.fit(arrs)
    preds = est.labels_.tolist()

    print('prediction {}:'.format(preds))
    print('gold value {}:'.format(indexes))
    print('Gold')
    print('*******')
    print_clusters(sentences, indexes)
    print('Predictions')
    print('*******')
    print_clusters(sentences, preds)
    # print('homogeneity: {}'.format(homogeneity_score(indexes, preds)))
    print('vi: {}'.format(v_measure_score(indexes, preds)))
    exit()
    hierchy(arrs)
    tsne(arrs)
    # hierchy(np.stack(arrs, axis=0))
