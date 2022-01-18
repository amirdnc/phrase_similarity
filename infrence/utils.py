import gzip
import json
import os
from collections import Set, Counter, defaultdict
import spacy
from nltk.corpus import stopwords
from tqdm import tqdm
import nltk

nlp = spacy.load("en_core_web_sm")


pronouns = 'I we you he she it they me us you her him it them mine ours yours hers his theirs my our your her his their myself yourself herself himself itself ourselves yourselves ' \
           'themselves all another any anybody anyone anything both each either everybody everyone everything few many most neither nobody none no one nothing one other others several some somebody someone something such'.lower().split(
    ' ')

data_dir = r"D:\reviews"

stop_words = set(stopwords.words('english'))
def load_json(path):
    """
    lod a json file
    :param path: path of input file
    :return: line iterator of the json object
    """
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)

def get_raw_data(data_filename):
    # data_dir = r'/home/nlp/amirdnc/data/reviews'
    in_path = os.path.join(data_dir, data_filename)
    data = [l['reviewText'] for l in load_json(in_path) if 'reviewText' in l]
    return data


def get_data_with_index(data_filename, number = 50000):
    data = get_raw_data(data_filename)
    if number:
        data = data[:number]
    indexs = defaultdict(list)
    indx_path = os.path.join(data_dir,'index_' + str(number) + '_' + data_filename + '.json')
    if os.path.exists(indx_path):
        with open(indx_path, 'r') as f:
            indexs = json.load(f)
        return indexs, data
    t_data = tqdm(data)
    for i, r in enumerate(t_data):
        if 'href' in r or 'http' in r:
            continue
        for x in nltk.word_tokenize(r.lower()):
            if x in stop_words or x in pronouns:
                continue
            indexs[x].append(i)
    with open(indx_path, 'w') as f:
        json.dump(indexs, f)
    return data, indexs

allowed_list = ['All_Beauty_5.json.gz', 'Arts_Crafts_and_Sewing_5.json.gz', 'Automotive_5.json.gz', 'Cell_Phones_and_Accessories_5.json.gz']
if __name__ == '__main__':
    for path in os.listdir(data_dir):
        if path.endswith('json'):
            continue
        if path not in allowed_list:
            continue
        print(path)
        get_data_with_index(path, 2000000)