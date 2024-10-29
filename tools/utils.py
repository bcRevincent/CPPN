# coding: utf-8

import logging
import torch
import torch.nn as nn
from collections import defaultdict, Counter
import torch.nn.functional as F
from openpyxl.styles.builtins import output, total
from torch.utils.tensorboard import SummaryWriter
import os
import shutil

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.handlers.clear()

    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def get_writer(config):
    if os.path.exists(config['writer_path']):
        shutil.rmtree(config['writer_path'])
    os.mkdir(config['writer_path'])
    writer = SummaryWriter(config['writer_path'])

    return writer


def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    log_mean_output = ((p_output + q_output) / 2).log()

    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2


def compute_loss_acc(prob, construct_output, batch_y, graph_loss_func, net, config):
    # cls_loss = graph_loss_func(prob[-1], batch_y.long())
    cls_loss = graph_loss_func(prob, batch_y.long())

    cls_acc = torch.eq(torch.argmax(prob[-1], dim=-1), batch_y.long()).float().mean().cpu().data

    construct_loss = F.mse_loss(construct_output[0], construct_output[1])

    if config['wo_ref']:
        total_loss = config['lambda_project_loss'] * construct_loss + config['lambda_cls_loss'] * cls_loss

        return total_loss, cls_acc

    inter_loss = js_div(net.semantic_edge + 1e-8, net.visual_edge + 1e-8)

    intra_loss = F.cross_entropy(net.visual_edge, torch.arange(net.semantic_edge.size(0)).to(config['device'])) + \
                 F.cross_entropy(net.semantic_edge, torch.arange(net.semantic_edge.size(0)).to(config['device']))

    # compute total loss
    total_loss = config['lambda_project_loss'] * construct_loss + config['lambda_cls_loss'] * cls_loss + \
                 config['lambda_inter_loss'] * inter_loss + config['lambda_intra_loss'] * intra_loss


    return total_loss, cls_acc