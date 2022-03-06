#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:40:16 2019

@author: weetee
"""
import platform
import random

import torch

from src.tasks.trainer import train_and_fit
from src.tasks.infer import infer_from_trained, FewRel
import logging
from argparse import ArgumentParser

'''
This fine-tunes the BERT model on SemEval, FewRel tasks
'''

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

seed_val = 1234
random.seed(seed_val)
# np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
if __name__ == "__main__":
    if 'Windows' in platform.platform():
        train_path = r"D:\reviews\Arts_Crafts_and_Sewing_5_triplet_train.json"
        dev_path = r"D:\reviews\Arts_Crafts_and_Sewing_5_triplet_dev.json"
        train_path = r"D:\reviews\Automotive_5_triplet_train.json"
        dev_path = r"D:\reviews\Automotive_5_triplet_dev.json"
        gold_eval = r"D:\human\automotive1500.csv"
        raw_eval = 'Automotive_5.json.gz'


        train_path = r"D:\reviews\Arts_Crafts_and_Sewing_5_triplet_train.json"
        dev_path = r"D:\reviews\Arts_Crafts_and_Sewing_5_triplet_dev.json"
        gold_eval = r"D:\human\art1500.csv"
        raw_eval = 'Arts_Crafts_and_Sewing_5.json.gz'
    else:
        train_path = r"../../data/reviews/Automotive_5_triplet_train.json"
        dev_path = r"../../data/reviews/Automotive_5_triplet_dev.json"
        gold_eval = r"/home/nlp/amirdnc/data/reviews/automotive1500.csv"
        raw_eval = 'Automotive_5.json.gz'


        train_path = r"../../data/reviews/Arts_Crafts_and_Sewing_5_triplet_train.json"
        dev_path = r"../../data/reviews/Arts_Crafts_and_Sewing_5_triplet_dev.json"
        gold_eval = r"/home/nlp/amirdnc/data/reviews/art1500.csv"
        raw_eval = 'Arts_Crafts_and_Sewing_5.json.gz'
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default='multi_noun_similarity', help='semeval, fewrel, noun_similarity', )
    # parser.add_argument("--task", type=str, default='double_negative_similarity', help='semeval, fewrel, noun_similarity', )
    parser.add_argument("--train_data", type=str,
                        default=train_path, \
                        help="training data .txt file path")
    parser.add_argument("--test_data", type=str,
                        default=dev_path, \
                        help="test data .txt file path")
    # parser.add_argument("--train_data", type=str,
    #                     default=r"D:\reviews\All_Beauty_5_triplet_train.json", \
    #                     help="training data .txt file path")
    # parser.add_argument("--test_data", type=str,
    #                     default=r"D:\reviews\All_Beauty_5_triplet_dev.json", \
    #                     help="test data .txt file path")

    parser.add_argument("--use_pretrained_blanks", type=int, default=0, help="0: Don't use pre-trained blanks model, 1: use pre-trained blanks model")
    parser.add_argument("--num_classes", type=int, default=19, help='number of relation classes')
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--gradient_acc_steps", type=int, default=2, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--fp16", type=int, default=0, help="1: use mixed precision ; 0: use floating point 32") # mixed precision doesn't seem to train well
    parser.add_argument("--num_epochs", type=int, default=11, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.00007, help="learning rate")
    parser.add_argument("--model_no", type=int, default=3, help='''Model ID: 0 - BERT\n
                                                                            1 - ALBERT\n
                                                                            2 - BioBERT''')
    parser.add_argument("--model_size", type=str, default='bert-base-uncased', help="For BERT: 'bert-base-uncased', \
                                                                                                'bert-large-uncased',\
                                                                                    For ALBERT: 'albert-base-v2',\
                                                                                                'albert-large-v2'\
                                                                                    For BioBERT: 'bert-base-uncased' (biobert_v1.1_pubmed)")
    parser.add_argument("--train", type=int, default=1, help="0: Don't train, 1: train")
    parser.add_argument("--infer", type=int, default=0, help="0: Don't infer, 1: Infer")
    parser.add_argument("--val_step", type=int, default=10, help="validation step")
    parser.add_argument('--model', type=str, default='SpanBERT/spanbert-base-cased', help='model string to use')
    parser.add_argument('--save_path', type=str, default='./data_art_base_once/',
                        help='path to save models and tokenizer')
    parser.add_argument('--test_path', type=str, default=gold_eval,
                        help='path for test')
    parser.add_argument('--test_raw', type=str, default=raw_eval,
                        help='path for raw test data')

    args = parser.parse_args()
    
    if args.train == 1:
        net = train_and_fit(args)

    if args.infer == 1:
        inferer = infer_from_trained(args, detect_entities=True)
        test = "The surprise [E1]visit[/E1] caused a [E2]frenzy[/E2] on the already chaotic trading floor."
        inferer.infer_sentence(test, detect_entities=False)
        test2 = "After eating the chicken, he developed a sore throat the next morning."
        inferer.infer_sentence(test2, detect_entities=True)

        while True:
            sent = input("Type input sentence ('quit' or 'exit' to terminate):\n")
            if sent.lower() in ['quit', 'exit']:
                break
            inferer.infer_sentence(sent, detect_entities=False)
