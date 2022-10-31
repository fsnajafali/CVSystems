import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, wide_resnet50_2

from typing import Callable

def pairwise_l2_distance(a: torch.Tensor, b: torch.Tensor):
    """
    Computes pairwise distances between all rows of a and all rows of b.
    :param a: tensor
    :param b: tensor
    :return pairwise distance
    """
    norm_a = torch.sum(torch.square(a), dim = 0)
    norm_a = torch.reshape(norm_a, [-1, 1])
    norm_b = torch.sum(torch.square(b), dim = 0)
    norm_b = torch.reshape(norm_b, [1, -1])
    a = torch.transpose(a, 0, 1)
    zero_tensor = torch.zeros(64, 64)
    dist = torch.maximum(norm_a - 2.0 * torch.matmul(a, b) + norm_b, zero_tensor)
    return dist


def get_sims(embs: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Calculates self-similarity between batch of sequence of embeddings
    :param embs: embeddings
    :param temperature: temperature
    :return self similarity tensor
    """

    batch_size = embs.shape[0]
    seq_len = embs.shape[2]
    embs = torch.reshape(embs, [batch_size, -1, seq_len])

    def _get_sims(embs: torch.Tensor):
        """
        Calculates self-similarity between sequence of embeddings
        :param embs: embeddings
        """

        dist = pairwise_l2_distance(embs, embs)
        sims = -1.0 * dist
        return sims

    sims = map_fn(_get_sims, embs)
    # sims = torch.Size[20, 64, 64]
    sims /= temperature
    sims = F.softmax(sims, dim = -1)
    sims = sims.unsqueeze(dim = -1)
    return sims


def map_fn(fn: Callable, elems: torch.Tensor) -> torch.Tensor:
    """
    Transforms elems by applying fn to each element unstacked on dim 0.
    :param fn: function to apply
    :param elems: tensor to transform
    :return: transformed tensor
    """

    sims_list = []
    for i in range(elems.shape[0]):
        sims_list.append(fn(elems[i]))
    sims = torch.stack(sims_list)
    return sims