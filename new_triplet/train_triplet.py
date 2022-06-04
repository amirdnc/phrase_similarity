import os
import pickle

import torch
from pytorch_metric_learning.losses import BaseMetricLossFunction
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from imports import get_trained_model
from new_triplet.evaluation import ppdb_embd_eval
from src.triplet import TrippletSpanModel, mini_triplet, combined_model
from pytorch_metric_learning import losses, miners, distances

import numpy as np

from wikipedia_similarity import eval_turney, ppdb_eval

CUDA_LAUNCH_BLOCKING=1

class triplet_data(Dataset):
    def __init__(self, data_dict, split_size):
        self.labels_to_int = {}
        self.samples = []
        self.labels = []
        for key, val in tqdm(data_dict.items()):
            if key in self.labels_to_int:
                print(f'duplicate key! {key}')
                continue
            val = torch.tensor(val)
            val = val.squeeze()
            if val.dim() == 1: # skip phrases with one instance
                continue
            self.labels_to_int[key] = len(self.labels_to_int)
            for features in torch.split(val, split_size):
                if features.squeeze().dim() == 1:  # don't take samples with size 1
                    # print("don't take samples with size 1")
                    continue
                self.samples.append(features)
                self.labels.append([self.labels_to_int[key]]*len(features))


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

def collate_wrapper(batch):
    features, labels = list(zip(*batch))
    sample = torch.cat(features)
    fin_labels =torch.tensor([item for sublist in labels for item in sublist])  # flatten list then convert to tensor
    return sample, fin_labels

def load_test_and_train(data, batch_size=50, test_precent=0.05):

    train = dict(list(data.items())[int(len(data) * test_precent):])
    test = dict(list(data.items())[:int(len(data) * test_precent)])
    train_ds = triplet_data(train, 10)
    test_ds = triplet_data(test, 10)
    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_wrapper, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_wrapper, shuffle=False)
    return train_loader, test_loader

def run_test(model, dataloader, loss_function):
    res = []
    print('starting test')
    model.eval()
    with torch.no_grad():
        for i, (samples, labels) in enumerate(dataloader):
            embeddings = model(samples.cuda())
            loss = loss_function(embeddings, labels)
            res.append(loss.cpu().detach().numpy())
            if i > 500:
                break
    print(f'test loss is {np.mean(res)}')
    model.train()
    return res


def normlize_labels(labels):
    labels = labels.numpy()
    convert = {x:y for x,y in zip(set(labels), range(len(set(labels))))}
    new_labels = [convert[x] for x in labels]
    return torch.tensor(new_labels)


def task_eval(model, data, prefix=''):
    ppdb_size = 15532
    # return ppdb_eval("D:\PPDB-filtered.json", 'SpanBERT/spanbert-base-cased', num_iterations=99999, start_index=ppdb_size*0.75, num_context=300, model=model)
    # return eval_turney(r"D:\turney.txt", 'SpanBERT/spanbert-base-cased', num_iterations=100, num_context=300, model=model)
    return ppdb_embd_eval("D:\PPDB-filtered.json", 'SpanBERT/spanbert-base-cased', data, prefix, num_iterations=99999, start_index=11649, model=model)

def main():
    model_name = 'SpanBERT/spanbert-base-cased'
    embbeding_size = 768
    lr = 0.0007
    output_folder = 'run_cosine1'
    data_path = r'D:\vec_dict.pickle'
    # data_path = r'D:\vec_dict_tiny.pickle'
    with open(data_path, 'rb') as handle:
        data = pickle.load(handle)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    emb_model = get_trained_model(model_name)

    model = mini_triplet(embbeding_size).cuda()
    test_model = combined_model(emb_model, model)
    train_loader, test_loader = load_test_and_train(data, batch_size=25)
    print(f'data len is {len(train_loader)} for train and {len(test_loader)} for test.')
    # loss_func = losses.CentroidTripletLoss()
    miner = miners.MultiSimilarityMiner()
    loss_func = losses.TripletMarginLoss(distance=distances.CosineSimilarity())
    optimizer = optim.Adam([{"params": model.parameters(), "lr": lr}])
    temp_losses = []
    epocs = 10
    results = []
    test_model.eval()
    # results.append(task_eval(model, data, output_folder +'/base'))
    test_model.train()
    best_res = 0
    for j in range(epocs):
        print(f'starting epoch {j}')
        model.train()
        for i, (samples, labels) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            embeddings = model(samples.cuda())
            hard_pairs = miner(embeddings, labels)
            # loss = loss_func(embeddings, labels)
            loss = loss_func(embeddings, labels, hard_pairs)
            temp_losses.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
            # if i % 1000 == 999:
            #     print(f'loss is {np.mean(temp_losses)}')
            #     print(f'test is {run_test(model, test_loader, loss_func)}')
            #     temp_losses = []
        # run_test(model, test_loader, loss_func)
        test_model.eval()
        results.append(task_eval(model, data, output_folder + f'/{j}'))
        test_model.train()
        torch.save(model.state_dict(), output_folder + r'/j.pt')
        if best_res < results[-1][0]:
            best_res = results[-1][0]
            torch.save(model.state_dict(), output_folder + r'/best_model.pt')
        print(f'Results so far: {results}')

if __name__ == '__main__':


    main()