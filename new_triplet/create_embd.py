import os
import pickle

from tqdm import tqdm
from transformers import AutoTokenizer

from imports import get_trained_model
from phrase_similarity import embd
from wikipedia_similarity import get_sentece_context


def main():
    data_path = r'D:\sentence_context'
    data_path = r'D:\Arts_Crafts_and_Sewing_10000/'
    model_name = r'SpanBERT/spanbert-base-cased'
    model = get_trained_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    d = {}
    files = os.listdir(data_path)
    for i, file in enumerate(tqdm(files)):
        phrase = file.replace('.csv', '').replace('_', ' ')
        phrases = get_sentece_context(phrase, data_dir=data_path)[:400]
        vectors = embd(phrases, phrase, tokenizer, model)
        d[phrase] = vectors.cpu().numpy()
        if i > len(files):
            break

    with open(r'D:\vec_dict_reviews.pickle', 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    main()