from random import shuffle

import torch
from torch import nn
from transformers import AutoModel, AutoConfig, AutoModelForMaskedLM


def norm_to_unit(v):
    qn = torch.norm(v, p=2, dim=1).detach()
    return v.div(qn.expand_as(v.T).T)


def dist_mat(mat1, mat2):
    mat1 = torch.stack([norm_to_unit(x) for x in mat1], dim=0)
    mat2 = torch.stack([norm_to_unit(x) for x in mat2], dim=0)
    return 1 - (torch.matmul(mat1, mat2.transpose(1, 2)))


def get_addtional_data(t):
    return {'input_ids': t, 'attention_mask': (t != 0).type(torch.int64).cuda(),
            'token_type_ids': torch.zeros_like(t).cuda()}


class TrippletModel(nn.Module):
    def __init__(self, name, margin=0.2):
        super(TrippletModel, self).__init__()
        # config = AutoConfig.from_pretrained(name)
        # self.model = AutoModel.from_config(config)
        self.model = AutoModelForMaskedLM.from_pretrained(name)
        self.loss = nn.TripletMarginLoss(margin=margin, p=2)
        self.margin = margin
        self.drop = nn.Dropout(p=0.5)

    def forward(self, input, index):
        prediction_scores = self.model(**get_addtional_data(input))[0]
        droped_scores = self.drop(prediction_scores)
        return droped_scores[torch.arange(droped_scores.size(0)), index]
        # return prediction_scores[torch.arange(prediction_scores.size(0)), index]

        # blankv1v2 = prediction_scores[0][:, index, :]
        # buffer = []
        # for i in range(blankv1v2.shape[0]):  # iterate batch & collect
        #     v1v2 = blankv1v2[i, i, :, :]
        #     v1v2 = torch.cat((v1v2[0], v1v2[1]))
        #     buffer.append(v1v2)
        # del blankv1v2
        # v1v2 = torch.stack([a for a in buffer], dim=0)
        # del buffer
        # return v1v2

    def run_train(self, pos0, pos1, neg, pos0_i, pos1_i, neg_i, get_embedding=False):
        p0 = self.forward(pos0, pos0_i)
        p1 = self.forward(pos1, pos1_i)
        n = self.forward(neg, neg_i)
        if get_embedding:
            return self.loss(p0, p1, n), p0, p1, n
        else:
            return self.loss(p0, p1, n)

    def run_train_multi(self, pos, neg, mask):
        ps = [self.forward(x, mask[0][:, j]) for (j, x) in enumerate(pos)]
        shuffle(ps)
        ns = [self.forward(x, mask[1][:, j]) for (j, x) in enumerate(neg)]
        pos_dist = dist_mat(ps, ps)
        neg_dist = dist_mat(ps, ns + ns)
        # neg_expns = torch.cat([neg_dist, neg_dist], dim=1)
        # neg_expns = torch.cat([neg_expns, neg_expns], dim=2)
        fin = torch.clamp(torch.sum(pos_dist - (neg_dist + torch.eye(neg_dist.size(1)).cuda()*self.margin) + self.margin), 0)
        loss = torch.sum(fin)
        # loss = torch.sum(torch.sum(fin, dim=2), dim=1)
        return loss


    def run_train_multi_reduce(self, pos, neg, mask):
        p_l = list(range(len(pos)))
        n_l = list(range(len(neg)))
        shuffle(p_l)
        shuffle(n_l)
        return self.run_train(pos[p_l[0]], pos[p_l[1]], neg[n_l[0]], mask[0][:, p_l[0]], mask[0][:, p_l[1]], mask[1][:, n_l[0]])


# class MultiTrippletModel(TrippletModel):
#     def __init__(self, name, margin=1.0):
#         super(MultiTrippletModel, self, name, margin).__init__()