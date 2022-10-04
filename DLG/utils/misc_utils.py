import collections.abc
import os
import cv2
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


def save_result(dir, pred, name):
    if not os.path.exists(dir):
        os.makedirs(dir)
    res = pred.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    cv2.imwrite(os.path.join(dir, str(name)), res * 255)

# def save_result(dir, pred, name):
#     if not os.path.exists(dir):
#         os.makedirs(dir)
#     res = pred.cpu().numpy().squeeze()
#     res = (res - res.min()) / (res.max() - res.min() + 1e-8)
#     cv2.imwrite(os.path.join(dir, str(name)), res * 255)











