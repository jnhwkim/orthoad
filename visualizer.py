from __future__ import absolute_import

import time
import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from model import *
from metric import *
from utils import *


class DataVisualizer(object):
    ''' Data visualizer using `matplotlib`.
    '''

    def __init__(self, data, post_fn, path=None):
        self.data = data
        self.n = 20
        self.col = 1

        self.post_fn = post_fn
        self.path = path
        self.fig, self.axes = plt.subplots(3, self.col, figsize=(12, 8))
        self.ax = self.axes.ravel()
        self.idx = 0
        self.threshold = 50.

        self.input_idx = list(range(self.col))
        self.gt_idx = list(range(self.col, self.col*2))
        self.output_idx = list(range(self.col*2, self.col*3))

        # Colorbar
        self.cbar_ax = self.fig.add_axes([0.93, 0.2, 0.02, 0.6])

        # Category Button
        if path is None:
            axbtn = plt.axes([0.15, .92, 0.06, 0.03])
            btn = Button(axbtn, 'Next')
            btn.on_clicked(self.next_sample)

        # Indexer
        self.text = plt.text(1.5, .3, 'indexer')

        self.visualize()

        if path is not None:
            self.save_pdf()

        plt.show()

    def visualize(self):
        for i, c in enumerate(self.data):
            if len(self.data[c]) == 3:
                inp, pred, gt = self.data[c]
                typ = None
            else:
                inp, pred, gt, typ, serial = self.data[c]
            for j in range(self.idx, self.idx + self.col):
                k = j % self.n
                if typ is not None:
                    title = '{}/{}'.format(serial[k], typ[k])
                    self.ax[self.input_idx[k % self.col]].set_title(title)
                self.ax[self.input_idx[k % self.col]].imshow(
                    chw2hwc(self.post_fn(inp[k])))
                outlined = DataVisualizer.outlined(self.post_fn(inp[k]), gt[k])
                self.ax[self.gt_idx[k % self.col]].imshow(chw2hwc(outlined))
                im = self.ax[self.output_idx[k % self.col]] \
                    .imshow(pred[k].clamp(0, 30),
                            aspect='auto', cmap=plt.get_cmap('jet'),
                            vmin=0, vmax=30)
            self.fig.colorbar(im, cax=self.cbar_ax)
            self.text.set_text('{}/{}'.format(self.idx, self.n))
            break

    def next_sample(self, e):
        self.idx = (self.idx + self.col) % self.n
        self.visualize()

    def threshold_update(self, val):
        self.threshold = val
        self.visualize()

    def save_pdf(self):
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages(self.path)
        t = self.n // self.col
        for _ in tqdm(range(t)):
            pp.savefig(self.fig)
            self.next_sample(None)
        pp.close()

    @staticmethod
    def outlined(x, mask, color_channel=1, padding=3, thickness=2, fill=1):
        outline = DataVisualizer.outline(mask, padding, thickness)
        outlined = x.clone()
        if 2 == len(outlined.size()):
            outlined[outline] = fill
        else:
            outlined[:, outline] = 0
            outlined[color_channel, outline] = fill
        return outlined

    @staticmethod
    def outline(mask, padding, thickness):
        assert padding > thickness
        outline = mask.clone().float()
        outline = nn.MaxPool2d(3, 1, 1)(outline.unsqueeze(0).unsqueeze(0))
        outlines = [outline]
        for _ in range(padding-1):
            outlines.append(nn.MaxPool2d(3, 1, 1)(outlines[-1]))
        outline = outlines[-1] == 1
        return outline.squeeze() & ~(outlines[-thickness-1].squeeze() == 1)


class SingleDataVisualizer(DataVisualizer):
    ''' Single data visualizer using `matplotlib`.
    '''

    def __init__(self, dataset, data, post_fn, args, path=None):
        self.dataset = dataset
        self.data = data
        self.n = args.num_samples if args.num_samples is not None \
            else data[0].size(0)
        self.col = 1

        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150

        self.post_fn = post_fn
        self.args = args
        self.path = path
        self.fig, self.ax = plt.subplots(1, 3, figsize=(16, 8))
        self.idx = 0

        # Category Button
        if path is None:
            axbtn = plt.axes([0.15, .92, 0.06, 0.03])
            btn = Button(axbtn, 'Next')
            btn.on_clicked(self.next_sample)

        self.data_b = None
        self.visualize()

        if path is not None:
            self.save_pdf()
            import subprocess, platform
            if 'Darwin' == platform.system():
                subprocess.call(['open', self.path])  # open pdf
        # plt.show()

    def visualize(self):
        x, y, a, c = self.dataset[self.idx]
        mask, pred, typ, serial = self.data
        if isinstance(c, torch.Tensor):
            c = c.item()

        # upscale if needed
        def resize_gray_img(x, size):
            return F.interpolate(x.unsqueeze(1), size=size)[:, 0]
        mask = resize_gray_img(mask, x.shape[1:])
        pred = resize_gray_img(pred, x.shape[1:])

        # update title
        title = serial[self.idx]
        self.ax[0].set_title(title)
        self.ax[2].set_title(
            '{} (green), predict (blue)'.format(
                typ[self.idx]))

        # update images
        self.ax[0].imshow(chw2hwc(self.post_fn(x)))

        # mask as outlines
        options = {'padding': 3, 'thickness': 1, 'fill': 1}
        gt_out = DataVisualizer.outlined(
            self.post_fn(x), a,
            color_channel=1, **options)
        pr_out = DataVisualizer.outlined(
            gt_out, mask[self.idx] > 0, color_channel=2, **options)

        self.ax[2].imshow(chw2hwc(pr_out))

        # prediction in jet
        sc_out = pred[self.idx]
        threshold = 15 if 'kolektor' == self.dataset.name \
            else 20

        if True:  # transparent colormap
            self.ax[1].imshow(chw2hwc(self.post_fn(x)))
            self.ax[1].imshow(sc_out.clamp(0, threshold),
                              aspect='equal', cmap=plt.get_cmap('jet'),
                              vmin=0, vmax=threshold, alpha=.5)
        else:
            self.ax[1].imshow(sc_out.clamp(0, threshold),
                              aspect='equal', cmap=plt.get_cmap('jet'),
                              vmin=0, vmax=threshold)

        for ax in self.ax:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

    def next_sample(self, e):
        self.idx = (self.idx + self.col) % self.n
        self.visualize()


if '__main__' == __name__:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--dataroot', default=os.environ['DATA'],
                        help='Path to the dataset')
    parser.add_argument('--dataset', default=MVTecAD.name,
                        help='Dataset to train')
    parser.add_argument('--split', default='test', help='Dataset test split')
    parser.add_argument('--val-split', default='val',
                        help='Dataset val split')
    parser.add_argument('--fold', type=int,
                        default=0, help='Cross validation fold')
    parser.add_argument('--category', default='all',
                        help='Dataset category to train')
    parser.add_argument('--batch-size', type=int,
                        default=1, help='Input batch size')
    parser.add_argument('--workers', type=int, default=0,
                        help='The number of workers for data loaders')
    parser.add_argument('--ckpt', default=None, type=str,
                        help='path to the trained model')
    parser.add_argument('--num_samples', default=10, type=int,
                        help='The number of samples to visualize')

    # default settings
    args = parser.parse_args()

    # logging
    logger = logging.getLogger(get_basename_without_ext(__file__))
    logger.info(greetings())
    logger.info(' '.join(os.sys.argv))
    logger.info(args)

    # data
    loaders = checkout_category_dataloaders(args)  # test
    for i, loader in enumerate(loaders):
        c = loader.dataset.category
        logger.info('For {:10}, nSamples={:4d}, nIters={:3d}'.format(
            c, *repr_loader(loader)))

    # visualization to pdf
    from visualizer import DataVisualizer
    loader = loaders[0]
    dataset = loader.dataset
    category = dataset.category
    path = get_path_with_new_file(args.ckpt, '{}.pth'.format(category))
    data = torch.load(path, map_location=torch.device('cpu'))
    path_pdf = get_path_with_new_file(args.ckpt, '{}.pdf'.format(category))
    SingleDataVisualizer(dataset, data, dataset.denorm, args, path_pdf)
    exit()
