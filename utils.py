import os
import math
import datetime
import logging
import time
import re
import copy
from tqdm import tqdm
import sys
from pathlib import Path
import numpy as np
import unittest
import functools
import operator

import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms import *
import cv2 as cv

""" J.A.R.V.I.S """


def greetings():
    currentTime = datetime.datetime.now()
    cheers = [', dude!', '!', '! How is today?',
              ', and good luck today!', ', good luck!']
    if currentTime.hour < 12:
        s = 'Good morning'
    elif 12 <= currentTime.hour < 18:
        s = 'Good afternoon'
    else:
        s = 'Good evening'
    idx = torch.Tensor(1).random_(len(cheers)).int().item()
    return s + cheers[idx]


""" Utility """


def inverse_list(list):  # list to dict: index -> element
    dict = {}
    for idx, x in enumerate(list):
        dict[x] = idx
    return dict


def subdirectory_names(path):
    return [x for x in os.listdir(path)
            if os.path.isdir(os.path.join(path, x))]


def test_subdirectory_names():
    assert 'wiki' in subdirectory_names('.'), 'include a subdirectory'
    assert 'utils.py' not in subdirectory_names('.'), 'exclude a file'


def assert_eq(a, b):
    assert abs(a - b) < 1e-9, '{} != {}'.format(a, b)


def repr_loader(loader):
    num_samples = len(loader.dataset)
    num_iters = math.ceil(num_samples / loader.batch_size)
    return num_samples, num_iters


def mkdirs(dirs):
    try:
        if isinstance(dirs, list):
            for d in dirs:
                os.makedirs(d)
        else:
            os.makedirs(dirs)
    except:
        pass


def get_basename_without_ext(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def get_path_with_new_file(path, filename):
    return os.path.join(os.path.os.path.dirname(path), filename)


def cache_func(fn, path, override=False, use_device=True, **kwargs):
    if kwargs.get('device', None) is None:
        kwargs['device'] = torch.device('cpu')
    device = kwargs['device']
    if os.path.isfile(path) and not override:
        return torch.load(path, map_location=device)
    else:
        if not use_device:
            kwargs.pop('device')
        ret = fn(**kwargs)
        torch.save(ret, path)
        return ret


def get_by_nearest(dict, x):
    nearest = min(dict.keys(), key=lambda k: abs(k-x))
    return dict[nearest], nearest


def pb_load(path, blocksize=1024, progressbar=tqdm):
    ''' numpy load with a progressbar
        Ref. https://stackoverflow.com/questions/42691876/load-npy-file-with-np-load-progress-bar
    '''
    try:
        mmap = np.load(path, mmap_mode='r')
        y = np.empty_like(mmap)
        n_blocks = int(np.ceil(mmap.shape[0] / blocksize))
        for b in progressbar(range(n_blocks)):
            y[b*blocksize: (b+1) * blocksize] = \
                mmap[b * blocksize: (b+1) * blocksize]
    finally:
        del mmap  # make sure file is closed again
    return y


""" Computer Vision """


def chw2hwc(x):
    return x.transpose(0, 1).transpose(1, 2)


def c2chw(x):
    return x.unsqueeze(1).unsqueeze(2)


def c2hwc(x):
    return x.unsqueeze(0).unsqueeze(1)


def watershed(x, mask, pooler=None, verbose=False):
    ''' Region-overlap counting with a revised Watershed algorithm using
            max-pooling for a batch of tensors.

        Args:
            x (Tensor): A marker-filled tensor with the size of (B, H, W).
                A marker is an unique integer to represent a color.
            mask (Tensor): A binary region mask with the size of (B, H, W).

        Ref:
            https://docs.opencv.org/3.4/d3/db4/tutorial_py_watershed.html
    '''
    t = time.time()
    if pooler is None:
        pooler = nn.MaxPool2d(3, 1, 1).to(x.device)
    it = 0
    while True:
        it += 1
        x_prev = x
        x = pooler(x.float()).long() * mask.long()
        if (x - x_prev).abs().sum().item() < 1e-9:
            break
        if it > x.size(1) * 2:
            torch.save(x, 'watershed_x.pth')
            torch.save(x_prev, 'watershed_x_prev.pth')
            assert (x - x_prev).abs().sum().item() < 1e-9, \
                (x - x_prev).abs().sum().item()
    if verbose:
        print('{:1.1f} elapsed for {}-iter watershed.'
              .format(time.time() - t, it))
    return x


def get_filter(size, shape='circle'):
    if 'circle' == shape:
        kernel = np.zeros((size, size), np.uint8)
        cv.circle(kernel, (size // 2, size // 2), size // 2, 1, -1)
        return kernel
    else:
        return np.ones((size, size), np.uint8)


def draw_rectangle(img: torch.Tensor, y, x, h, w,
                   thickness=2, color=[1, 1, 1]):
    y_ = max(0, y - thickness)
    y__ = min(img.size(1), y + h + thickness)
    x_ = max(0, x - thickness)
    x__ = min(img.size(2), x + w + thickness)
    for c in range(3):
        # top
        img[c, y_:y, x_:x__] = color[c]
        # bottom
        img[c, y + h:y__, x_:x__] = color[c]
        # left
        img[c, y_:y__, x_:x] = color[c]
        # right
        img[c, y_:y__, x + w:x__] = color[c]


def save_png_from_array(img: np.ndarray, filename):
    assert not filename.endswith('.png'), 'extension `png` would be added.'
    if img is None:
        return False
    if 2 == len(img.shape):
        im = Image.fromarray(np.uint8(img * 255), 'L')
        im.save(filename + '.png')
    elif 3 == len(img.shape):
        im = Image.fromarray(img, 'RGB')
        im.save(filename + '.png')
    elif 4 == len(img.shape):
        for i in range(img.shape[0]):
            im = Image.fromarray(np.transpose(img[i], (1, 2, 0)), 'RGB')
            im.save(filename.format(i) + '.png')
    return True


""" Deep Learning """


def generic_forward(module, x):
    if isinstance(module, nn.ModuleList):
        output = []
        if isinstance(x, list) and len(x) == len(module):
            for m, x_ in zip(module, x):
                output.append(m(x_))
        else:
            for m in module:
                output.append(m(x))
        return output
    else:
        return module(x)


def module_clone(module):
    return copy.deepcopy(module)


def num_params_trainable(model):
    return num_params(model, lambda p: p.requires_grad)


def num_params(model, fn=lambda x: True):
    nParams = 0
    for w in filter(fn, model.parameters()):
        nParams += functools.reduce(operator.mul, w.size(), 1)
    return nParams


def module_list_clone(list, fill=None):
    if fill is None:
        return [x.clone() for x in list]
    else:
        return [x.clone().fill_(fill) for x in list]


def color_normalize(x):
    if isinstance(x, np.ndarray):
        for i in range(x.shape[0]):
            x[i] = (x[i] - np.mean(x[i])) / (np.std(x[i]))
        return x
    else:
        if 4 == len(x.size()):
            b, c, h, w = x.size()
            x = x.view(b, c, -1)
            x = (x - x.mean(2, keepdim=True)) / (x.std(2, keepdim=True) + 1e-9)
            return x.view(b, c, h, w)
        elif 3 == len(x.size()):
            c, h, w = x.size()
            x = x.view(c, -1)
            x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-9)
            return x.view(c, h, w)


def input_complement(x, mode='train', method='random'):
    def apply(x, mask):
        factor = x.size(1) / mask.float().sum().item()
        mask = mask.to(x.device)
        return x * mask.long() * factor
    if 'random' == method:
        if 'train' != mode:
            return x, x
        mask = torch.randperm(x.size(1)) < 1
        mask = mask.unsqueeze(0)
        for i in range(len(x.size())-2):
            mask = mask.unsqueeze(-1)
        return apply(x, mask), x
    elif 'separate' == method:
        mask = torch.BoolTensor([True, True, True])
        mask[0] = False
        mask = mask.unsqueeze(0)
        for i in range(len(x.size())-2):
            mask = mask.unsqueeze(-1)
        
        return apply(x, mask), apply(x, ~mask)
    elif 'gray' == method:
        return x.mean(dim=1, keepdim=True).repeat(1,3,1,1), x


def tuple_eq(tup, x):
    def fn(x, y): return x == y
    return tuple_check(tup, x, fn)


def tuple_gt(tup, x):
    def fn(x, y): return x > y
    return tuple_check(tup, x, fn)


def tuple_check(tup, x, fn):
    if isinstance(tup, tuple):
        return all([fn(t, x) for t in tup])
    else:
        return fn(tup, x)


def flatten(x, dim=0):
    if 0 != dim:
        x = x.transpose(0, dim)
    return x.reshape(x.size(0), -1)


""" Experiments """


def set_logging_config(logdir, filename='log.txt'):
    handlers = [logging.FileHandler(os.path.join(logdir, filename)),
                logging.StreamHandler(os.sys.stdout)]
    for handler, level in zip(handlers, [logging.INFO, logging.INFO]):
        handler.setLevel(level)
    logging.basicConfig(format="[%(asctime)s][%(name)s] %(message)s",
                        level=logging.INFO, handlers=handlers)


def parse_log(path, regex, label, types):
    ''' Parse log file into a dictionary

        Args:
            path   (str): r"logs/mvtecad_grid_conv65_remove_last_relu_1/log.txt"
            regex  (str): '\[(.*)\]\[(.*)\] \[Epoch\s*(.+)\] \[Time\s*(.+)\] \[Val\s*(.+)\]'
            label (list): ['time', 'name', 'epoch', 'elapse', 'val']
            types (list): [str, str, int, float, float]
    '''
    with open(path, "r") as file:
        match_list = {}
        for line in file:
            for match in re.finditer(regex, line, re.S):
                for i, l in enumerate(label):
                    if match_list.get(l, None) is None:
                        match_list[l] = []
                    match_list[l].append(types[i](match[i+1]))
    file.close()
    return match_list


def next_file(path, postfix='_'):
    # return postfix numbering increased by one (e.g., file -> file_[1-99999])
    # warning: path should be exclude postfix
    root = os.path.dirname(path)
    file_list = os.listdir(root)
    MAX_INDEX = 99999
    for i in range(1, MAX_INDEX):
        c_path = '{}_{}'.format(path, i)
        if os.path.basename(c_path) not in file_list:
            return c_path
    raise NotImplementedError('too many indices! ({})'.format(MAX_INDEX))


def test_next_file():
    assert './wiki_1' == next_file('./wiki')


def get_default_experiment(args):
    if hasattr(args, 'multiscale') and args.multiscale:
        label = '_multiscale' + args.label
    else:
        label = args.label
    f = '{}_{}_{}{}'.format(
        args.dataset, args.category, args.model, label)
    return next_file(os.path.join('logs', f))


def time_to_human(s):
    out = []
    s = int(s)
    u = [60, 60, 24, 99999]
    for i in range(len(u)):
        out.insert(0, s % u[i])
        if s < u[i]:
            break
        s = math.floor(s / u[i])
    return out


def sec_to_str(s):
    t = time_to_human(s)
    f = ['{:2d}d', '{:2d}h', '{:2d}m', '{:2d}s']
    if 1 <= len(t) <= 4:
        return ' '.join(f[4-len(t):]).format(*t)
    else:
        raise NotImplementedError()


""" Hand-on examples """


def test_random_perspective():
    image = Image.open(
        '../../dataset/mvtec_anomaly_detection/metal_nut/train/good/000.png',
        'r')
    t1 = RandomPerspective(distortion_scale=0.1, p=0.999)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes.ravel()
    ax[0].imshow(image)
    ax[1].imshow(t1(image))
    plt.show()


if __name__ == '__main__':  # simple testcases
    test_subdirectory_names()
    test_next_file()
    test_random_perspective()
