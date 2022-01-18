import os
import sys

import torch
from torch import optim

import logging
import os
cwd = os.getcwd()

sys.path.insert(1, cwd)
from src.triplet import TrippletModel

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)
def load_state(net, optimizer, scheduler,  load_best=False, base_path="./data/"):
    """ Loads saved model and optimizer states if exists """
    amp_checkpoint = None
    checkpoint_path = os.path.join(base_path,"task_test_checkpoint_%d.pth.tar" % 3)
    best_path = os.path.join(base_path,"task_test_model_best_%d.pth.tar" % 3)
    start_epoch, best_pred, checkpoint = 0, 0, None
    if (load_best == True) and os.path.isfile(best_path):
        checkpoint = torch.load(best_path)
        logger.info("Loaded best model.")
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        logger.info("Loaded checkpoint model.")
    if checkpoint != None:
        start_epoch = checkpoint['epoch']
        best_pred = checkpoint['best_acc']
        net.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        amp_checkpoint = checkpoint['amp']
        logger.info("Loaded model and optimizer.")
    return start_epoch, best_pred, amp_checkpoint


def get_trained_model(name, path="./data/"):
    net = TrippletModel(name)
    net.cuda()

    optimizer = optim.Adam([{"params": net.parameters(), "lr": 0.0007}])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 6, 8, 12, 15, 18, 20, 22, \
                                                                      24, 26, 30], gamma=0.8)

    start_epoch, best_pred, amp_checkpoint = load_state(net, optimizer, scheduler, load_best=False, base_path=path)
    return net