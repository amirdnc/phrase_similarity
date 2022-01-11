import logging
import random
from argparse import ArgumentParser

import torch
from torch import optim
from tqdm import tqdm
from transformers import AutoModel, AutoModelForMaskedLM
import  numpy as np
from src.tasks.preprocessing_funcs import load_dataloaders
from src.tasks.train_funcs import l2, load_state
from src.triplet import TrippletModel

logger = logging.getLogger('__file__')

seed_val = 1234
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
def run_sbert(model, input):
    attention_mask = ((input==0)*1-1)*-1
    model_output = model(input)
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_num_best(mat, i):
    return np.sum(mat==i, axis=0)
def calc_all(p0, p1, n):
    best = []
    for i in range(len(p0)):
        for j in range(len(p1)):
            for k in range(len(n)):
                res = []
                x, y, z = l2(p0[i], p1[j]), l2(n[k], p1[j]), l2(p0[i], n[k])
                res.append(torch.logical_and(x < y, x < z))
                res.append(torch.logical_and(y < x, y < z))
                res.append(torch.logical_and(z < y, z < x))
                v = torch.argmax(torch.stack(res).to(int), dim=0).detach().cpu().numpy()
                best.append(v)
    # print(best)
    mat = np.stack(best)
    final = np.argmax(np.stack([get_num_best(mat, 0), get_num_best(mat, 1), get_num_best(mat, 2)]), axis=0)
    return sum(final==0)

def get_trained_model(args):
    net = TrippletModel(args.model)
    net.cuda()

    optimizer = optim.Adam([{"params": net.parameters(), "lr": args.lr}])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 6, 8, 12, 15, 18, 20, 22, \
                                                                      24, 26, 30], gamma=0.8)

    start_epoch, best_pred, amp_checkpoint = load_state(net, optimizer, scheduler, args, load_best=False)
    return net

def eval_sebert(test_loader, model=None, limit=10):
    # tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
    if not model:
        net = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens").cuda()
    else:
        net = model
    if isinstance(net, torch.nn.DataParallel):
        net = net.module

    logger.info("Evaluating test samples...")
    tot_loss = 0; tot = 0; good = 0
    net.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):

            p, n, n2, masks = data #todo fix evaluation
            # p0 = p0.cuda();
            p = [x.cuda() for x in p]
            n = [x.cuda() for x in n]
            # ps = [run_sbert(net, x) for x in p]
            # ns = [run_sbert(net, x) for x in n]
            ps = [net.forward_aux(x, masks[0][:,j]).detach() for (j,x) in enumerate(p[:limit*2])]
            ns = [net.forward_aux(x, masks[1][:,j]).detach() for (j,x) in enumerate(n[:limit])]
            p1 = ps[:int(len(ps)/2)]
            p0 = ps[int(len(ps)/2):]

            # p1 = [ps[0]]
            # p0 = [ps[1]]
            # ns = [ns[0]]
            # goods = torch.logical_and(l2(p0, p1) < l2(p0, n), l2(p0, p1) < l2(p1, n))
            # good += sum(goods).cpu().numpy()
            good += calc_all(p0, p1, ns)
            tot += p0[0].size(0)
            # tot += len(goods)
            cur_loss = net.run_train_multi(p, n, masks)
            if torch.cuda.device_count() > 1:
                cur_loss = cur_loss.sum()
            tot_loss += cur_loss
    accuracy = good / tot
    loss = tot_loss/(i + 1)
    results = {
        "accuracy": accuracy,
        "loss": loss
    }
    logger.info("***** Eval results *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
    return results

if __name__ == "__main__":
    test_path = r"D:\reviews\Arts_Crafts_and_Sewing_5_triplet_dev.json"
    test_path = r"D:\reviews\All_Beauty_5_triplet_dev.json"

    train_path = r"D:\reviews\Arts_Crafts_and_Sewing_5_triplet_train.json"
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default='multi_noun_similarity', help='semeval, fewrel, noun_similarity', )
    parser.add_argument("--train_data", type=str,
                        default=train_path, \
                        help="training data .txt file path")
    parser.add_argument("--test_data", type=str,
                        default=test_path, \
                        help="test data .txt file path")
    # parser.add_argument("--train_data", type=str,
    #                     default=r"D:\reviews\All_Beauty_5_triplet_train.json", \
    #                     help="training data .txt file path")
    # parser.add_argument("--test_data", type=str,
    #                     default=r"D:\reviews\All_Beauty_5_triplet_dev.json", \
    #                     help="test data .txt file path")

    parser.add_argument("--use_pretrained_blanks", type=int, default=0, help="0: Don't use pre-trained blanks model, 1: use pre-trained blanks model")
    parser.add_argument("--num_classes", type=int, default=19, help='number of relation classes')
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
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
    parser.add_argument("--val_step", type=int, default=30000, help="validation step")
    parser.add_argument('--model', type=str, default='SpanBERT/spanbert-base-cased', help='model string to use')


    args = parser.parse_args()
    train_loader, test_loader, train_len, test_len = load_dataloaders(args, True)
    model = get_trained_model(args)
    # model = AutoModelForMaskedLM.from_pretrained(args.model)
    eval_sebert(test_loader, model, limit=1)