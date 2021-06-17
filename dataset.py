from __future__ import absolute_import

import os
import math
import matplotlib.pyplot as plt
import numpy as np
import unittest
from PIL import Image
from tqdm import tqdm
import pickle
from glob import glob

import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import ImageFolder, ImageNet
import torchvision.transforms as T
from torchvision.datasets.folder import pil_loader
import cv2 as cv

import random
from random import sample, shuffle

from utils import *


def checkout_dataloader(args, splits=['train', 'val', 'test']):
    loaders = []

    def get_dataset(args, split):
        if 'all' == args.category:
            dataset = MVTecAD.merge_categories(args.dataroot, split)
        else:
            assert args.category in MVTecAD.categories
            dataset = MVTecAD(args.dataroot, split, args.category)
        return dataset
    if MVTecAD.name == args.dataset:
        if 'train' in splits:
            loaders.append(torch.utils.data.DataLoader(
                get_dataset(args, 'train'),
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=not hasattr(args, 'drop_last') or args.drop_last,
                num_workers=args.workers
            ))
        if 'val' in splits:
            loaders.append(torch.utils.data.DataLoader(
                get_dataset(args, 'val'),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers
            ))
        if 'test' in splits:
            loaders.append(torch.utils.data.DataLoader(
                get_dataset(args, 'test'),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers
            ))
    elif KolektorSDD.name == args.dataset:
        for split in splits:
            drop_last = ('test' != split) and \
                (not hasattr(args, 'drop_last') or args.drop_last)
            loaders.append(torch.utils.data.DataLoader(
                KolektorSDD(args.dataroot, 
                            fold=args.fold, split=split, scale=args.scale,
                            negative_only='train' == split),
                batch_size=args.batch_size,
                shuffle='train' == split,
                drop_last=drop_last,
                num_workers=args.workers
            ))
    elif KolektorSDD2.name == args.dataset:
        for split in splits:
            drop_last = ('test' != split) and \
                (not hasattr(args, 'drop_last') or args.drop_last)
            loaders.append(torch.utils.data.DataLoader(
                KolektorSDD2(args.dataroot, 
                            split=split, scale=args.scale,
                            negative_only='train' == split),
                batch_size=args.batch_size,
                shuffle='train' == split,
                drop_last=drop_last,
                num_workers=args.workers
            ))
    elif STCAD.name == args.dataset:
        if 'train' in splits:
            loaders.append(torch.utils.data.DataLoader(
                STCAD(dataroot=args.dataroot, split='train'),
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=not hasattr(args, 'drop_last') or args.drop_last,
                num_workers=args.workers
            ))
        if 'val' in splits:
            loaders.append(torch.utils.data.DataLoader(
                STCAD(dataroot=args.dataroot, split='val'),
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=not hasattr(args, 'drop_last') or args.drop_last,
                num_workers=args.workers
            ))
        if 'test' in splits:
            loaders.append(torch.utils.data.DataLoader(
                STCAD(dataroot=args.dataroot, split='test',
                      start_idx=args.startidx),
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=not hasattr(args, 'drop_last') or args.drop_last,
                num_workers=args.workers
            ))
    return loaders


def checkout_category_dataloaders(args):
    split = args.split
    if MVTecAD.name == args.dataset:
        loaders = []
        if 'test' == split:
            if 'all' == args.category:
                categories = MVTecAD.categories
            else:
                categories = [args.category]
            for c in categories:
                loaders.append(torch.utils.data.DataLoader(
                    MVTecAD(args.dataroot, 'test', category=c),
                    batch_size=args.batch_size,
                    shuffle='train' == split,
                    num_workers=args.workers
                ))
        else:
            raise NotImplementedError()
        return loaders
    elif KolektorSDD.name == args.dataset:
        loaders = []
        if 'test' == split:
            print(args.fold)
            loaders.append(torch.utils.data.DataLoader(
                KolektorSDD(args.dataroot, 
                            fold=args.fold if hasattr(args, 'fold') else 0, 
                            split=split, scale=args.scale \
                                if hasattr(args, 'scale') else 'half',
                            negative_only='train' == split),
                batch_size=args.batch_size,
                shuffle='train' == split,
                num_workers=args.workers
            ))
        else:
            raise NotImplementedError()
        return loaders
    elif KolektorSDD2.name == args.dataset:
        loaders = []
        if 'test' == split:
            print(args.fold)
            loaders.append(torch.utils.data.DataLoader(
                KolektorSDD2(args.dataroot, 
                            split=split, scale=args.scale \
                                if hasattr(args, 'scale') else 'half',
                            negative_only='train' == split),
                batch_size=args.batch_size,
                shuffle='train' == split,
                num_workers=args.workers
            ))
        else:
            raise NotImplementedError()
        return loaders
    elif STCAD.name == args.dataset:
        loaders = []
        if 'test' == split:
            loaders.append(torch.utils.data.DataLoader(
                STCAD(args.dataroot, split=split,
                      start_idx=args.startidx),
                batch_size=args.batch_size,
                shuffle='train' == split,
                num_workers=args.workers
            ))
        else:
            raise NotImplementedError()
        return loaders


def checkout_output_size(args):
    if MVTecAD.name == args.dataset:
        return 256


def checkout_input_size(args):
    if MVTecAD.name == args.dataset:
        return 256, 256


class MVTecAD(ImageFolder):
    ''' MVTec Anomaly Detection dataset

    This expoits `ImageFolder`, but this can build an aggregated dataset over
        texture or object categories with consistent classes.

        Args:
            dataroot (string): path to the root directory of the dataset
            split    (string): data split ['train', 'val', 'test']
            category (string): a product type in `MVTecAD.categories`
    '''
    name = 'mvtecad'

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    categories = [
        'carpet', 'grid', 'leather', 'tile', 'wood',
        'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
        'pill', 'screw', 'toothbrush', 'transistor', 'zipper'
    ]

    labels = [  # cumulative unique labels
        'good',  # common negative label
        'color', 'cut', 'hole', 'metal_contamination', 'thread',  # carpet
        'bent', 'broken', 'glue',  # grid
        'fold', 'poke',  # leather
        'crack', 'glue_strip', 'gray_stroke', 'oil', 'rough',  # tile
        'combined', 'liquid', 'scratch',  # wood
        'broken_large', 'broken_small', 'contamination',  # bottle
        'bent_wire', 'cable_swap', 'cut_inner_insulation', \
        'cut_outer_insulation', \
        'missing_cable', 'missing_wire', 'poke_insulation',  # cable
        'faulty_imprint', 'squeeze',  # capsule
        'print',  # hazelnut
        'flip',  # metal_nut
        'pill_type',  # pill
        'manipulated_front', 'scratch_head', 'scratch_neck', \
        'thread_side', 'thread_top',  # screw
        'defective',  # toothbrush
        'bent_lead', 'cut_lead', 'damaged_case', 'misplaced',  # transistor
        'broken_teeth', 'fabric_border', 'fabric_interior', \
        'split_teeth', 'squeezed_teeth'  # zipper
    ]

    def __init__(self,
                 dataroot='../machine_vision/dataset/mvtec_anomaly_detection',
                 split='train', category=None):
        assert category is not None, 'argument `category` is required!'
        super(MVTecAD, self).__init__(
            root=os.path.join(dataroot, category,
                              'train' if 'val' == split else split),
            loader=self.cache_loader,
            transform=self.get_transform(split)
        )

        class_to_idx = inverse_list(self.labels)

        if 'train' == split or 'val' == split:
            samples = []
            for idx, s in enumerate(self.samples):
                # ten-percent validation
                if ('val' == split and 0 == idx % 10) or \
                   ('train' == split and 0 != idx % 10):
                    samples.append(
                        (s[0], class_to_idx[self.classes[s[1]]], None))
                self.samples = samples
        else:
            annotations = ImageFolder(
                os.path.join(dataroot, category, 'ground_truth'),
                loader=self.cache_loader,
                transform=self.get_transform(split)
            )
            j = 0
            for idx, s in enumerate(self.samples):
                if 'good' == self.classes[s[1]]:
                    self.samples[idx] = (
                        s[0], class_to_idx[self.classes[s[1]]], None)
                    continue
                else:
                    a = annotations.samples[j]
                    j += 1
                assert self.classes[s[1]] == annotations.classes[a[1]], \
                    'inconsistent label {} != {}'.format(s[1], a[1])
                self.samples[idx] = (
                    s[0], class_to_idx[self.classes[s[1]]], a[0])

        self.split = split
        self.category = category
        self.targets = [s[1] for s in self.samples]
        self.classes = self.labels
        self.class_to_idx = class_to_idx
        self.cam = 0

        image_cache_path = 'cache/.mvtecad_{}_{}'.format(category, split)
        if os.path.isfile(image_cache_path):
            self.image_cache = torch.load(image_cache_path)
        else:
            self.image_cache = {}
            self.preprocess()
            print('Image cache saved.')
            torch.save(self.image_cache, image_cache_path)

        self.transform = T.Compose([
            self.transform,
            T.Normalize(MVTecAD.mean, MVTecAD.std)
        ])

    def __getitem__(self, index):
        path, target, path_a = self.samples[index]
        sample = self.loader(path, self.image_cache)
        if path_a is not None:
            annotation = self.loader(path_a, self.image_cache)
        else:
            annotation = torch.Tensor(1)  # dummy not used

        ''' After making cache, `mean` key exists. Before that, do not
                transform to get the statistics over images, mean and std.
        '''
        if self.transform is not None and 'mean' in self.image_cache.keys():
            sample = self.transform(sample)
            if path_a is not None:  # annotation is one-channel
                annotation = self.transform(annotation)[0] > .5

        if self.target_transform is not None:
            target = self.target_transform(target)

        if path_a is None and 'mean' in self.image_cache.keys():
            annotation = torch.zeros_like(sample[0]) > .5

        return sample, target, annotation, 0

    def preprocess(self):
        def to_tensor(x): return T.ToTensor()(x)
        for i in range(len(self)):
            data = self[i]
            sample = to_tensor(data[0])
            if 0 == i:
                mean = sample.clone()
                count = 1
            else:
                mean += sample
                count += 1
        mean /= count
        std = torch.zeros_like(mean)
        for i in range(len(self)):
            data = self[i]
            sample = to_tensor(data[0])
            std += (sample - mean)**2
        std /= count
        std = std ** .5
        self.image_cache['mean'] = mean
        self.image_cache['std'] = std

    @staticmethod
    def merge_categories(dataroot, split):
        datasets = []
        for c in MVTecAD.categories:
            datasets.append(MVTecAD(dataroot, split, category=c))
        return ConcatDataset(datasets)

    @staticmethod
    def cache_loader(path, cache_dict):
        img = cache_dict.get(path, None)
        if img is None:
            pretrain_transform = MVTecAD.get_transform(mode='pretrain')
            img = pretrain_transform(pil_loader(path))
            cache_dict[path] = img
        return img

    @staticmethod
    def get_transform(mode='pretrain'):
        if 'pretrain' == mode:
            return T.Compose([
                T.Resize(320),  # if h > w, then rescaled to (x*h/w, x)
            ])
        elif 'train' == mode:
            return T.Compose([
                T.Resize(256),
                T.RandomCrop(256),
                # T.RandomGrayscale(p=0.1),
                T.ToTensor()
            ])
        elif 'test' == mode or 'val' == mode:
            return T.Compose([
                T.Resize(256),
                T.CenterCrop(256),
                T.ToTensor()
            ])
        else:
            raise NotImplementedError()

    @staticmethod
    def denorm(x):
        return x * c2chw(torch.Tensor(MVTecAD.std)) + \
            c2chw(torch.Tensor(MVTecAD.mean))


class KolektorSDD(Dataset):
    ''' Kolektor Surface-Defect dataset

        Args:
            dataroot (string): path to the root directory of the dataset
            fold     (int)   : fold in [0, 1, 2]
            split    (string): data split ['train', 'test']
            scale    (string): input image scale
            debug    (bool)  : debug mode
    '''
    name = 'kolektor'
    category = 'commutator'
    labels = ['ok', 'defect']
    scales = {'1408x512': 1., '704x256': .5, 'half': .5}  # height x width
    output_sizes = {'1408x512': (1408, 512), 
                    '704x256': (704, 256), 
                    'half': (704, 256)}  # height

    # ImageNet
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(self,
                 dataroot='/path/to/dataset/'
                          'KolektorSDD',
                 fold=0, split='train', scale='half', negative_only=False, 
                 debug=False):
        super(KolektorSDD, self).__init__()

        self.dataroot = dataroot
        self.split_path = dataroot + '-training-splits/split.pyb'
        self.fold = fold
        self.split = split
        self.scale = scale
        self.fxy = self.scales[scale]
        self.output_size = self.output_sizes[scale]
        self.negative_only = negative_only
        self.debug = debug
        
        self.class_to_idx = inverse_list(self.labels)
        self.classes = self.labels
        self.transform = KolektorSDD.get_transform(output_size=self.output_size)
        self.normalize = T.Normalize(KolektorSDD.mean, KolektorSDD.std)

        image_cache_path = 'cache/.kolektor_{}_{}'.format(split, fold)
        if os.path.isfile(image_cache_path):
            self.samples, self.masks, self.product_ids = \
                torch.load(image_cache_path)
        else:
            self.load_imgs()
            torch.save((self.samples, self.masks, self.product_ids), 
                       image_cache_path)
        if negative_only:
            m = self.masks.sum(-1).sum(-1) == 0
            self.samples = self.samples[m]
            self.masks = self.masks[m]
            self.product_ids = [pid for flag, pid in zip(m, self.product_ids) 
                                    if flag]
    def load_imgs(self):
        with open(self.split_path, 'rb') as f:
            [train_split, test_split, all] = pickle.load(f)
        
        targets = train_split if 'train' == self.split else test_split
        N = len(targets[self.fold]) * 8
        if 'kos21' in targets[self.fold]:
            N -= 1

        self.samples = torch.Tensor(N, *self.output_size).zero_()
        self.masks = torch.LongTensor(N, *self.output_size).zero_()
        self.product_ids = []

        cnt = 0
        for product_id in targets[self.fold]:
            # warning: inconsistent ordering across macOS and linux
            image_list = glob("%s/%s/*.jpg" % (self.dataroot, product_id))
            assert 0 < len(image_list), self.dataroot 
            for img_name in image_list:
                img = self.transform(Image.open(img_name).convert('L'))
                lab = self.transform(Image.open(
                    img_name[:-4] + '_label.bmp').convert('L'))
                self.samples[cnt] = img
                self.masks[cnt] = lab
                self.product_ids.append(product_id + '_' + img_name[-5:-4])
                cnt += 1

    def __getitem__(self, index):
        x = torch.stack([self.samples[index]] * 3, 0)
        a = self.masks[index] > 0
        if self.normalize is not None:
            x = self.normalize(x)

        if 0 == a.sum():
            y = self.class_to_idx['ok']
        else:
            y = self.class_to_idx['defect']

        return x, y, a, 0

    def __len__(self):
        return self.samples.size(0)

    @staticmethod
    def get_transform(mode='preprocess', output_size=(1408, 512)):
        transform = [
            T.Resize(output_size),
            T.ToTensor()
        ]
        transform = T.Compose(transform)
        return transform

    @staticmethod
    def denorm(x):
        return x * c2chw(torch.Tensor(KolektorSDD.std)) + \
            c2chw(torch.Tensor(KolektorSDD.mean))


class KolektorSDD2(Dataset):
    ''' Kolektor Surface-Defect 2 dataset

        Args:
            dataroot (string): path to the root directory of the dataset
            split    (string): data split ['train', 'test']
            scale    (string): input image scale
            debug    (bool)  : debug mode
    '''
    name = 'kolektor2'
    category = 'commutator'
    labels = ['ok', 'defect']
    scales = {'1408x512': 1., '704x256': .5, 'half': .5}  # height x width
    output_sizes = {'1408x512': (1408, 512), 
                    '704x256': (704, 256), 
                    'half': (704, 256)}  # height

    # ImageNet
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(self,
                 dataroot='/path/to/dataset/'
                          'KolektorSDD2',
                 split='train', scale='half', negative_only=False, 
                 debug=False):
        super(KolektorSDD2, self).__init__()

        self.dataroot = dataroot
        self.split_path = None
        self.fold = None
        self.split = 'train' if 'val' == split else split
        self.scale = scale
        self.fxy = self.scales[scale]
        self.output_size = self.output_sizes[scale]
        self.negative_only = negative_only
        self.debug = debug
        
        self.class_to_idx = inverse_list(self.labels)
        self.classes = self.labels
        self.transform = KolektorSDD.get_transform(output_size=self.output_size)
        self.normalize = T.Normalize(KolektorSDD.mean, KolektorSDD.std)
        
        image_cache_path = 'cache/.kolektor2_{}'.format(split)
        if os.path.isfile(image_cache_path):
            self.samples, self.masks, self.product_ids = \
                torch.load(image_cache_path)
        else:
            self.load_imgs()
            torch.save((self.samples, self.masks, self.product_ids), 
                       image_cache_path)
        if negative_only:
            m = self.masks.sum(-1).sum(-1) == 0
            self.samples = self.samples[m]
            self.masks = self.masks[m]
            self.product_ids = [pid for flag, pid in zip(m, self.product_ids) 
                                    if flag]
    def load_imgs(self):
        # Please remove the duplicated files in the official dataset:
        #   10301_GT (copy).png, 10301 (copy).png
        N = 2331 if 'train' == self.split else 1004

        self.samples = torch.Tensor(N, 3, *self.output_size).zero_()
        self.masks = torch.LongTensor(N, *self.output_size).zero_()
        self.product_ids = []

        cnt = 0
        path = "%s/%s/" % (self.dataroot, self.split)
        image_list = [f for f in os.listdir(path) 
                      if re.search(r'[0-9]+\.png$', f)]
        assert 0 < len(image_list), self.dataroot 

        for img_name in image_list:
            product_id = img_name[:-4]
            img = self.transform(Image.open(path + img_name))
            lab = self.transform(
                Image.open(path + product_id + '_GT.png').convert('L'))
            self.samples[cnt] = img
            self.masks[cnt] = lab
            self.product_ids.append(product_id)
            cnt += 1

        assert N == cnt, '{} should be {}!'.format(cnt, N) 

    def __getitem__(self, index):
        x = self.samples[index]
        a = self.masks[index] > 0
        if self.normalize is not None:
            x = self.normalize(x)

        if 0 == a.sum():
            y = self.class_to_idx['ok']
        else:
            y = self.class_to_idx['defect']

        return x, y, a, 0

    def __len__(self):
        return self.samples.size(0)

    @staticmethod
    def get_transform(mode='preprocess', output_size=(704, 256)):
        transform = [
            T.Resize(output_size),
            T.ToTensor()
        ]
        transform = T.Compose(transform)
        return transform

    @staticmethod
    def denorm(x):
        return x * c2chw(torch.Tensor(KolektorSDD.std)) + \
            c2chw(torch.Tensor(KolektorSDD.mean))


class STCAD(ImageFolder):
    ''' ShanghaiTech Campus Dataset
    
        Args:
            dataroot  (string): path to the root directory of the dataset
            split     (string): data split ['train', 'test']
            start_idx    (int): start number of frame when subsampling 
                                with factor of 5 in test split (5n + ith)
    '''
    name = 'mstc'
    category = 'surveilance'
    labels = ['normal', 'anomaly']

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    
    def __init__(self,
                 dataroot='/path/to/dataset/STC',
                 split='train', start_idx=0):
        assert split in ['train', 'val', 'test']
        super(STCAD, self).__init__(
            root=os.path.join(dataroot, 'converted', 'train' if split == 'val' else split),
            loader=self.cache_loader,
            transform=self.get_transform(split)
        )
        
        label_dict = inverse_list(self.labels)

        if split == 'train' or split == 'val':
            self.samples = sample(self.samples, 5000)

            samples = []

            for idx, s in enumerate(self.samples):
                if (split == 'val' and 0 == idx % 10) or (split == 'train' and 0 != idx % 10):
                    samples.append((s[0], label_dict['normal'], None))

            self.samples = samples
            
        elif split == 'test':
            self.samples = self.subsample_test_split(start_idx)
            sample_idx = 0
            
            scenes = os.listdir(F"{dataroot}/converted/{split}")
            scenes.sort()
            
            for i, scene in enumerate(scenes):
                fm = np.load(os.path.join(F"{dataroot}/archive/test_frame_mask/{scene}.npy"))[start_idx::5]
                
                for _i, _fm in enumerate(fm):
                    if _fm == label_dict['normal']:
                        path_a = None
                    elif _fm == label_dict['anomaly']:
                        assert self.samples[sample_idx + _i][0].split('/')[-1][:-4] == \
                            F"{_i * 5 + start_idx:03}", 'Inconsistant index between ' + \
                            F"input image {self.samples[sample_idx + _i]} " + \
                            F" and pixel mask {scene}/{_i * 5 + start_idx:03}.jpg"
                        path_a = F"{dataroot}/converted/pixel_mask/{scene}/{_i * 5 + start_idx:03}.jpg"
                    else:
                        raise ValueError('Unexceptable label')

                    self.samples[sample_idx + _i] = (self.samples[sample_idx + _i][0], _fm, path_a)
                sample_idx += len(fm)

        else:
            raise ValueError("split must be one of 'train', 'valid' or 'test'.")

        self.split = split
        self.targets = [s[1] for s in self.samples]
        self.classes = self.labels

        image_cache_path = 'cache/.stcad_{}'.format(split)
        if os.path.isfile(image_cache_path):
            self.image_cache = torch.load(image_cache_path)
        else:
            self.image_cache = {}
            self.preprocess()
            print('Image cache saved.')
            torch.save(self.image_cache, image_cache_path)
            
        self.transform = T.Compose([
            self.transform,
            T.Normalize(STCAD.mean, STCAD.std)
        ])
        
    def __getitem__(self, index):
        path, target, path_a = self.samples[index]
        sample = self.loader(path, self.image_cache)
            
        if path_a is not None:
            annotation = self.loader(path_a, self.image_cache)
        else:
            annotation = torch.Tensor(1)  # dummy not used

        ''' After making cache, `mean` key exists. Before that, do not
                transform to get the statistics over images, mean and std.
        '''
        if self.transform is not None and 'mean' in self.image_cache.keys():
            sample = self.transform(sample)

            if path_a is not None:  # annotation is one-channel
                annotation = STCAD.get_transform(self.split)(annotation)[0] > 0 # bicubic makes value < 1
                annotation = annotation.float() # bool to float (0 or 1)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if path_a is None and 'mean' in self.image_cache.keys():
            annotation = torch.zeros_like(sample[0])

        return sample, target, annotation, 0

    def preprocess(self):
        def to_tensor(x): return T.ToTensor()(x)
        for i in range(len(self)):
            data = self[i]
            sample = to_tensor(data[0])
            if 0 == i:
                mean = sample.clone()
                count = 1
            else:
                mean += sample
                count += 1
        mean /= count
        std = torch.zeros_like(mean)
        for i in range(len(self)):
            data = self[i]
            sample = to_tensor(data[0])
            std += (sample - mean)**2
        std /= count
        std = std ** .5
        self.image_cache['mean'] = mean
        self.image_cache['std'] = std
                
    @staticmethod
    def cache_loader(path, cache_dict={}):
        img = cache_dict.get(path, None)
        if img is None:
            pre_transform = STCAD.get_transform('pretrain')
            img = pre_transform(pil_loader(path))
            cache_dict[path] = img
        
        return img
    
    @staticmethod
    def get_transform(mode='pretrain'):
        if 'pretrain' == mode:
            return T.Compose([
                T.Resize((256, 256)),  # if h > w, then rescaled to (x*h/w, x)
            ])
        elif 'train' == mode:
            return T.Compose([
                T.Resize((256, 256)),
                T.ToTensor()
            ])
        elif 'test' == mode or 'val' == mode:
            return T.Compose([
                T.Resize((256, 256)),
                T.ToTensor()
            ])
        else:
            raise NotImplementedError()

    def subsample_test_split(self, start_idx = 0):
        return [img_path for img_path in self.samples \
                if (int(img_path[0].split('/')[-1][:-4]) % 5 == start_idx)]
