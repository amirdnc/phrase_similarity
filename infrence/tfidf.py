import json
import os
from collections import defaultdict

from nltk import sent_tokenize
# from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from tqdm import tqdm

from utils import get_data_with_index, get_raw_data

nlp = English()
# Create a blank Tokenizer with just the English vocab
tokenizer = nlp.tokenizer

def calc_idf(docs):
    idf = defaultdict(int)
    for d in tqdm(docs):
        # for s in sent_tokenize(d):
        ws = set(x.text for x in tokenizer(d))
        for w in ws:
            idf[w] += 1
    idf['total_size'] = len(docs)
    return idf

def get_idf(path, data = None):
    out_path = path.replace('.json.gz', '_idf.json')
    if os.path.exists(out_path):
        with open(out_path) as f:
            idf = json.load(f)
            return idf
    # indexes, docs = get_data_with_index("All_Beauty_5.json.gz")
    if data != None:
        docs = data
    else:
        docs = get_raw_data("All_Beauty_5.json.gz")
    idf = calc_idf(docs)
    with open(out_path, 'w') as f:
        json.dump(idf, f)
    return idf

def avrg_idf(s, idf):
    ws = [x.text for x in tokenizer(s) if x.text in idf]
    if len(ws) == 0:
        return 0
    return sum(idf['total_size']/idf[w] for w in ws)/len(ws)

if __name__ == '__main__':
    path = "All_Beauty_5.json.gz"
    path = 'Cell_Phones_and_Accessories_5.json.gz'
    # path = 'Arts_Crafts_and_Sewing_5.json.gz'
    idf = get_idf(path)
    print(idf)
