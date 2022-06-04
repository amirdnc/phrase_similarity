import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from src.triplet import mini_triplet, TrippletModel, combined_model


def draw_sample(p1, p2, p1_embd, p2_embd, dist, label, save_location='', ind1=None, ind2=None):
    # print(f'p1: {p1}, p2: {p2}')
    # print(f'dist: {dist}, label: {label}.')

    p1_size = p1_embd.size(0)
    all_vec = torch.cat([p1_embd, p2_embd]).cpu().detach().numpy().T
    pca = PCA(n_components=2)
    pca.fit(all_vec)
    # print(pca.components_)
    p1_dots = pca.components_[:,:p1_size].T
    p2_dots = pca.components_[:,p1_size:].T

    if ind1 is not None:
        p1_dots = p1_dots[ind1]
    if ind2 is not None:
        p2_dots = p2_dots[ind2]
    if len(p2_dots.shape) == 1:
        p2_dots = np.expand_dims(p2_dots, axis=0)

    plt.cla()
    plt.scatter(p1_dots[:,0], p1_dots[:,1],  c='red', alpha=0.5, label=p1)
    plt.scatter(p2_dots[:,0], p2_dots[:,1],  c='blue', alpha=0.5, label=p2)
    plt.plot([], [], ' ', label=f'dist: {dist}, label: {label}.')
    plt.legend()
    # plt.show()
    # if not os.path.exists(save_location):
    #     os.mkdir(save_location)
    plt.savefig(save_location + f'{p1} - {p2}.png')
    return 1

def get_model(name):
    net = TrippletModel(name)
    embbeding_size = 768
    model_path = r"C:\Users\Amir\Dropbox\workspace_python\phrase_similarity\new_triplet\run_cosine1\best_model.pt"
    model = mini_triplet(embbeding_size)
    model.load_state_dict(torch.load(model_path))
    final_model = combined_model(net, model)
    return final_model

