from __future__ import absolute_import

import math
import unittest
import types
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import _resnet, BasicBlock, Bottleneck

from dataset import *
from metric import *
from utils import *

from scipy.ndimage import gaussian_filter


def checkout_model(args, hint=None):
    ''' Checkout a model instance based on training options.
    '''
    if 'resnet18' == args.model:
        model = resnet18(pretrained=True)
        model.inspect = types.MethodType(pixel_inspect, model)

    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    ''' 65 -> 32 -> 16 -> 8 -> 4
    '''
    model = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                    **kwargs)
    model.fc.__init__(model.fc.in_features, 1)  # fc hotfix
    return model


class TestResNet18(unittest.TestCase):
    def test_resnet18(self):
        from utils import assert_eq
        model = resnet18(True)
        # assert 1 ==  and 512 == model.fc.weight.shape[1]
        self.assertEqual(1, model.fc.weight.shape[0])
        self.assertEqual(512, model.fc.weight.shape[1])
        assert_eq(0.05759342014789581,
                  model.layer1[0].conv1.weight[0, 0, 0, 0].item())
        x = torch.Tensor(1, 3, 65, 65)
        self.assertIsNotNone(model(x))


class Mul2d(nn.Module):
    offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]

    def __init__(self, module, stochastic=False):
        super(Mul2d, self).__init__()
        assert tuple_eq(module.stride, 2)
        self.module = module
        self.stochastic = stochastic
        if hasattr(module, 'weight'):
            self.weight = module.weight

    def forward(self, x):
        if self.stochastic:
            idx = (torch.rand(1) * len(self.offsets)).long()
            offset = self.offsets[idx]
            x_ = F.pad(x, (offset[1], offset[1], offset[0], offset[0]))
            return self.module(x_)[:, :, offset[0]:, offset[1]:]
        else:
            outputs = []
            for offset in self.offsets:
                x_ = F.pad(x, (offset[1], offset[1], offset[0], offset[0]))
                outputs.append(self.module(x_)[:, :, offset[0]:, offset[1]:])
            return torch.cat(outputs, dim=0)


class MulMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MulMaxPool2d, self).__init__()
        assert kernel_size == 2 and stride == 2
        self.maxpool1 = nn.MaxPool2d(kernel_size, stride, (0, 0))
        self.maxpool2 = nn.MaxPool2d(kernel_size, stride, (0, 1))
        self.maxpool3 = nn.MaxPool2d(kernel_size, stride, (1, 0))
        self.maxpool4 = nn.MaxPool2d(kernel_size, stride, (1, 1))

    def forward(self, x):
        out = []
        out.append(self.maxpool1(x))
        out.append(self.maxpool2(x)[:, :, :, 1:])
        out.append(self.maxpool3(x)[:, :, 1:, :])
        out.append(self.maxpool4(x)[:, :, 1:, 1:])
        return torch.cat(out, dim=0)


class MulUnPool2d(nn.Module):
    def __init__(self, stride):
        super(MulUnPool2d, self).__init__()
        assert stride == 2
        self.stride = stride

    def forward(self, x):
        b, c, h, w = x.size()
        assert b % self.stride**2 == 0
        x = x.view(self.stride, self.stride, int(b / self.stride**2), c, h, w)
        x = x.transpose(0, 2)  # (r1,c1,b,c,r2,c2) -> (b,c1,r1,c,r2,c2) : mv b
        x = x.transpose(1, 3)  # (b,c1,r1,c,r2,c2) -> (b,c,r1,c1,r2,c2) : mv c
        # (b,c,r1,c1,r2,c2) -> (b,c,r2,r1,c1,c2) : mv r2
        x = x.transpose(3, 4).transpose(2, 3)
        x = x.transpose(4, 5)  # (b,c,r2,r1,c1,c2) -> (b,c,r2,r1,c2,c1) : mv c2
        return x.reshape(-1, c, 2 * h, 2 * w)


class TestMulMaxPool2d(unittest.TestCase):
    def test_mulmaxpool2d(self, verbose=False):
        '''
        This snippet shows that iterative pixel-wise convolutions with max-pooling
            can be inferred by a single forwarding.
            No-padding example since the padding complicates the problem.
        '''

        # data
        x = torch.Tensor(3, 14, 14).normal_().abs()

        # conv
        conv1 = nn.Conv2d(3, 3, 3, padding=0)
        conv2 = nn.Conv2d(3, 1, 3, padding=0)

        # net
        net = nn.Sequential(
            conv1,  # 10 -> 8
            nn.MaxPool2d(2, 2),  # 8 -> 4
            conv2,  # 4 -> 2
            nn.MaxPool2d(2, 2),  # 2 -> 1
        )

        # baseline b
        p = 10  # receptive field size
        b = torch.Tensor(5, 5).zero_()
        for i in range(5):
            for j in range(5):
                crop = x[:, i:i+p, j:j+p]
                out = net(crop.unsqueeze(0))
                b[i, j] = out.squeeze()
        if verbose:
            print(b)

        # proposed a
        mynet = nn.Sequential(
            conv1,  # 14 -> 12
            MulMaxPool2d(2, 2),  # 12 -> 6
            conv2,  # 6 -> 4
            MulMaxPool2d(2, 2),  # 4 -> 2
        )

        a = mynet(x.unsqueeze(0))
        c = a.clone()
        a = a.squeeze()  # 16 x 2 x 2
        a = a.view(2, 2, 2, 2, 2, 2)
        # transpose dimensions r1c1r0c0r2c2 -> r2r1r0c2c1c0
        a = a.transpose(1, 4)
        a = a.transpose(0, 1).transpose(3, 5)
        # exclude out-of-field
        a = a.reshape(8, 8)[:5, :5]
        if verbose:
            print(a)

        self.assertTrue((a-b).abs().mean() < 1e-9,
                        '{}'.format((a-b).abs().mean()))

        # MulUnPool2d test
        c = MulUnPool2d(2)(c)
        c = MulUnPool2d(2)(c)
        c = c[0, 0][:5, :5]

        self.assertTrue((b-c).abs().mean() < 1e-9,
                        '{}'.format((b-c).abs().mean()))


class Conv65(nn.Module):
    def __init__(self, bias=False, in_channels=3, out_channels=512,
                 hid_channels=128, expansion=1, slope=5e-3):
        super(Conv65, self).__init__()
        self.expansion = E = expansion
        self.conv1 = nn.Conv2d(in_channels,  int(128*E), 5, bias=bias)
        self.conv2 = nn.Conv2d(int(128*E),   int(128*E), 5, bias=bias)
        self.conv3 = nn.Conv2d(int(128*E),   int(128*E), 5, bias=bias)
        self.conv4 = nn.Conv2d(int(128*E),   int(256*E), 4, bias=bias)
        self.conv5 = nn.Conv2d(int(256*E), hid_channels, 1, bias=bias)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.relu = nn.LeakyReLU(slope, True)
        self.fc = nn.Linear(128, out_channels)
        self.moduleList = [
            self.conv1, self.relu, self.maxpool,
            self.conv2, self.relu, self.maxpool,
            self.conv3, self.relu, self.maxpool,
            self.conv4, self.relu,
            self.conv5]

    def forward(self, x):
        for i, m in enumerate(self.moduleList):
            x = m(x)
        return torch.flatten(x, 1)  # collapse


class Conv33(Conv65):
    def __init__(self, bias=False, in_channels=3, out_channels=256,
                 hid_channels=128, expansion=1, slope=5e-3):
        super(Conv33, self).__init__(slope=slope)
        self.expansion = E = expansion
        self.conv1 = nn.Conv2d(in_channels,  int(128*E), 5, bias=bias)
        self.conv2 = nn.Conv2d(int(128*E),   int(256*E), 5, bias=bias)
        self.conv3 = nn.Conv2d(int(256*E),   int(256*E), 2, bias=bias)
        self.conv4 = nn.Conv2d(int(256*E), hid_channels, 4, bias=bias)
        # to match with the output size of stage 3
        self.fc = nn.Linear(128, out_channels)
        self.moduleList = [
            self.conv1, self.relu, self.maxpool,
            self.conv2, self.relu, self.maxpool,
            self.conv3, self.relu,
            self.conv4, CenterCrop2d(1)]
        del self.conv5  # from Conv65


class Conv17(Conv65):
    def __init__(self, bias=False, in_channels=3, out_channels=128,
                 hid_channels=128, expansion=1, slope=5e-3):
        super(Conv17, self).__init__(slope=slope)
        self.expansion = E = expansion
        self.conv1 = nn.Conv2d(in_channels,  int(128*E), 5, bias=bias)
        self.conv2 = nn.Conv2d(int(128*E),   int(256*E), 5, bias=bias)
        self.conv3 = nn.Conv2d(int(256*E),   int(256*E), 5, bias=bias)
        self.conv4 = nn.Conv2d(int(256*E), hid_channels, 5, bias=bias)
        # to match with the output size of stage 2
        self.fc = nn.Linear(128, out_channels)
        self.moduleList = [
            self.conv1, self.relu,
            self.conv2, self.relu,
            self.conv3, self.relu,
            self.conv4, CenterCrop2d(1)]
        del self.conv5  # from Conv65
        del self.maxpool


class Conv9(Conv65):
    def __init__(self, bias=False, in_channels=3, out_channels=128,
                 hid_channels=128, expansion=1, slope=5e-3):
        super(Conv9, self).__init__(slope=slope)
        self.expansion = E = expansion
        self.conv1 = nn.Conv2d(in_channels,  int(128*E), 3, bias=bias)
        self.conv2 = nn.Conv2d(int(128*E),   int(256*E), 3, bias=bias)
        self.conv3 = nn.Conv2d(int(256*E),   int(256*E), 3, bias=bias)
        self.conv4 = nn.Conv2d(int(256*E), hid_channels, 3, bias=bias)
        # to match with the output size of stage 2
        self.fc = nn.Linear(128, out_channels)
        self.moduleList = [
            self.conv1, self.relu,
            self.conv2, self.relu,
            self.conv3, self.relu,
            self.conv4, CenterCrop2d(1)]
        del self.conv5  # from Conv65
        del self.maxpool


class KDConv33(nn.Module):
    def __init__(self, module, branch=False, label=''):
        super(KDConv33, self).__init__()
        self.module = module
        self.branch = branch
        self.fast_dense = False
        self.stochastic = False
        self.label = label
    
        if branch:
            def make_branch(in_channels, out_channels):
                return nn.Sequential(
                    DatasetNorm2d(in_channels))
            self.branches = nn.ModuleList([
                make_branch(128, 128),
                make_branch(256, 256),
                make_branch(256, 256),
                make_branch(128, 128)
                ])
        else:
            self.branches = None

        self.init()

        del self.module

    def init(self):
        self.moduleList = nn.ModuleList([
            nn.Sequential(self.module.conv1), 
            nn.Sequential(self.module.relu, 
                          self.module.maxpool,
                          self.module.conv2), 
            nn.Sequential(self.module.relu, 
                          self.module.maxpool,
                          self.module.conv3), 
            nn.Sequential(self.module.relu,
                          self.module.conv4)])

        self.pooling_count = 2

    def fastdense(self, stochastic=False):
        def transform_resnet(m, namespace=[], visited=[]):
            pooling_count = 0
            for n, c in m.named_children():
                if (isinstance(c, nn.Conv2d) or \
                    isinstance(c, nn.MaxPool2d) or \
                    isinstance(c, nn.AvgPool2d)) \
                    and hasattr(c, 'stride') \
                    and tuple_gt(c.stride, 1):
                    print('\t{} {} {} replaced with {}'.format(
                       ' '.join(namespace), n, 
                       type(c).__name__, 'Mul2d'))
                    setattr(m, n, Mul2d(c, stochastic=stochastic))
                    if 'downsample' not in namespace:
                        pooling_count += 1
                else:
                    if c not in visited and not isinstance(c, Mul2d):
                        pooling_count += transform_resnet(
                            c, namespace + [n], visited + [c])
            return pooling_count

        self.pooling_count = transform_resnet(self.moduleList)
        self.fast_dense = True
        self.stochastic = stochastic

        self.pooling_counts = [0, 1, 2, 2]
        self.center_crop_size = [256, 258, 260, 260]
        self.receptive_sizes = [5, 15, 21, 33]

    def forward(self, x, w=None, detach=False):
        if self.fast_dense:
            output_size = (x.size(-2), x.size(-1))
            p = 16
            x = F.pad(x, (p, p, p, p))

        outputs = []

        for i, b in enumerate(self.moduleList):
            if w is not None and b == self.moduleList[-1]:
                x = self.moduleList[-1][0](x)
                if detach:
                    x = x.detach()
                x = F.conv2d(x, w[-1])
            else:
                x = b(x)
                outputs.append(x)    

        if self.fast_dense and not self.stochastic:
            for i in range(len(outputs)):
                # hotfix
                self.center_crop_size[i] = \
                    (output_size[0] + 2 * self.pooling_counts[i], 
                     output_size[1] + 2 * self.pooling_counts[i])
                for j in range(self.pooling_counts[i]):
                    outputs[i] = MulUnPool2d(2)(outputs[i])
                outputs[i] = CenterCrop2d(self.center_crop_size[i])(outputs[i])
                outputs[i] = LeftTopCrop2d(output_size)(outputs[i])

        return outputs


class DatasetNorm2d(nn.Module):
    def __init__(self, num_features):
        super(DatasetNorm2d, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.count = 0
        self.track_running_stats = False

    def forward(self, x):
        if self.track_running_stats:
            if self.running_mean is None:
                self.running_mean += flatten(x, dim=1).mean(-1)
                self.running_var += flatten(x, dim=1).var(-1)
                self.count += x.size(0)
            else:
                n = self.count + x.size(0)
                self.running_mean.mul_(self.count / n)
                self.running_mean += \
                    (x.size(0) / n) * flatten(x, dim=1).mean(-1)
                self.running_var.mul_(self.count / n)
                self.running_var += \
                    (x.size(0) / n) * flatten(x, dim=1).var(-1)
                self.count += x.size(0)
        return (x - self.unsqueeze(self.running_mean)) / \
            self.unsqueeze(self.running_var + 1e-9).pow(.5)

    def unsqueeze(self, x):
        return x.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)


class DataNorm2d(torch.nn.BatchNorm2d):
    def forward(self, input):
        b, c, h, w = input.size()
        input_ = input.reshape(b, c * h * w, 1, 1)
        input_ = torch.cat([input_]*2, dim=0)
        output = super(DataNorm2d, self).forward(input_)[:1]
        return output.view(b, c, h, w)


class MultiscaleResNet(nn.Module):
    def __init__(self, module, preReLU=False):
        super(MultiscaleResNet, self).__init__()
        self.module = module
        self.avgpool = nn.Sequential(
            CenterCrop2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.preReLU = preReLU
        if preReLU:
            self.relu = nn.ReLU(inplace=False)
            for l in [self.module.layer2,
                      self.module.layer3,
                      self.module.layer4]:
                self.remove_last_relu(l)

    def forward(self, x):
        x = self.module.conv1(x)
        x = self.module.bn1(x)
        x = self.module.relu(x)
        x = self.module.maxpool(x)

        x = f1 = self.module.layer1(x)
        x = f2 = self.module.layer2(x)
        if self.preReLU:
            x = self.relu(x)
        x = f3 = self.module.layer3(x)
        if self.preReLU:
            x = self.relu(x)
        x = f4 = self.module.layer4(x)

        return [
            torch.flatten(self.avgpool(f4), 1),
            torch.flatten(self.avgpool(f3), 1),
            torch.flatten(self.avgpool(f2), 1)]

    def remove_last_relu(self, layer):
        layer[-1] = MyBasicBlock(layer[-1])


class MultiscaleIdentity(nn.Module):
    def __init__(self, out_channels=None):
        super(MultiscaleIdentity, self).__init__()
        self.out_channels = out_channels
        if self.out_channels is not None:
            self.fc = nn.Conv2d(3, self.out_channels, 1, 1)

    def forward(self, x):
        if self.out_channels is not None:
            x = self.fc(x)
        return [x] * 3


class MyBasicBlock(nn.Module):
    def __init__(self, module):
        super(MyBasicBlock, self).__init__()
        self.module = module
        self.moduleList = [
            self.module.conv1, self.module.bn1, self.module.relu,
            self.module.conv2, self.module.bn2
        ]

    def forward(self, x):
        identity = x

        if self.module.downsample is not None:
            identity = self.module.downsample(x)

        for m in self.moduleList:
            x = m(x)

        x += identity
        return x


class LeftTopCrop2d(nn.Module):
    def __init__(self, crop_size, offset=0):
        super(LeftTopCrop2d, self).__init__()
        self.crop_size = crop_size if isinstance(crop_size, tuple) else \
            (crop_size, crop_size)
        self.offset = offset

    def forward(self, x):
        return x[:, :,
                 self.offset:self.offset+self.crop_size[0],
                 self.offset:self.offset+self.crop_size[1]]

    def extra_repr(self):
        s = 'crop_size={crop_size}'
        return s.format(**self.__dict__)


class CenterCrop2d(nn.Module):
    def __init__(self, crop_size):
        super(CenterCrop2d, self).__init__()
        self.crop_size = crop_size if isinstance(crop_size, tuple) else \
            (crop_size, crop_size)

    def forward(self, x):
        row_pos = int(math.floor((x.size(2) - self.crop_size[0])/2))
        col_pos = int(math.floor((x.size(3) - self.crop_size[1])/2))
        return x[:, :,
                 row_pos: row_pos + self.crop_size[0],
                 col_pos: col_pos + self.crop_size[1]]

    def extra_repr(self):
        s = 'crop_size={crop_size}'
        return s.format(**self.__dict__)


class RandomCrop2d(nn.Module):
    def __init__(self, crop_size):
        super(RandomCrop2d, self).__init__()
        self.crop_size = int(crop_size)

    def forward(self, x):
        row_pos = math.floor(torch.rand(1) * (x.size(2) - self.crop_size + 1))
        col_pos = math.floor(torch.rand(1) * (x.size(3) - self.crop_size + 1))
        return x[:, :,
                 row_pos: row_pos + self.crop_size,
                 col_pos: col_pos + self.crop_size
                 ]

    def extra_repr(self):
        s = 'crop_size={crop_size}'
        return s.format(**self.__dict__)


class FastDense(nn.Module):
    def __init__(self, module, padding=0, output_size=256):
        super(FastDense, self).__init__()
        self.module = module
        self.padding = padding
        self.output_size = output_size
        self.moduleList = self._transform()
        self.fc = module.fc

    def _transform(self):
        moduleList = []
        pooling_count = 0
        dn = None
        for m in self.module.moduleList:
            if isinstance(m, nn.MaxPool2d):
                moduleList.append(MulMaxPool2d(m.kernel_size, m.stride))
                pooling_count += 1
                assert m.stride == 2
            elif isinstance(m, CenterCrop2d):
                pass
            elif isinstance(m, DataNorm2d):
                dn = m
            else:
                moduleList.append(m)
        for i in range(pooling_count):
            moduleList.append(MulUnPool2d(2))

        moduleList.append(LeftTopCrop2d(self.output_size))
        if dn is not None:
            moduleList.append(dn)

        return moduleList

    def forward(self, x):
        p = self.padding
        # x = F.pad(x, (p, p, p, p), 'reflect')
        x = F.pad(x, (p, p, p, p))
        for m in self.moduleList:
            x = m(x)
        return x

    def update_padding(self, padding):
        self.padding = padding


class MultiFastDense(nn.Module):
    def __init__(self, module, padding=(32, 16, 8), output_size=256):
        super(MultiFastDense, self).__init__()
        assert isinstance(module, nn.ModuleList)
        self.module = nn.ModuleList()
        self.padding = padding
        for i, m in enumerate(module):
            self.module.append(
                FastDense(m, padding=padding[i], output_size=output_size))

    def forward(self, x):
        return generic_forward(self.module, x)

    def update_padding(self, padding):
        for i, m in enumerate(self.module):
            m.update_padding(padding[i])


class TestFastDense(unittest.TestCase):
    def setUp(self):
        self.p = 65
        self.x = torch.rand(1, 3, 256, 256)
        self.net0 = Conv65()
        self.net1 = FastDense(self.net0, output_size=-1)
        self.out0 = torch.Tensor(1, 128, 256-self.p, 256-self.p).zero_()
        self.cuda()

    def cuda(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.x = self.x.to(device)
        self.out0 = self.out0.to(device)
        self.net0 = self.net0.to(device)
        self.net1 = self.net1.to(device)

    def test_no_padding(self):
        # for-loop
        p = self.p  # receptive field size
        out0 = self.out0
        s = 10
        for i in tqdm(range(s)):  # range(256-p)):
            for j in range(s):
                crop = self.x[:, :, i:i+p, j:j+p]
                out0[:, :, i, j] = self.net0(crop)
        # print(out0[0,0,:s,:s])
        # fastdense
        out1 = self.net1(self.x)
        # print(out1[0,0,:s,:s])
        self.assertTrue(
            (out0[0, 0, :s, :s]-out1[0, 0, :s, :s]).abs().mean().item() < 1e-5,
            '{}'.format(
                (out0[0, 0, :s, :s]-out1[0, 0, :s, :s]).abs().mean().item()))


class SpadeResNet(nn.Module):
    def __init__(self, module, preReLU=False, label=''):
        super(SpadeResNet, self).__init__()
        self.module = module
        self.preReLU = preReLU or ('preReLU' in label)
        self.fast_dense = False
        self._debug = False
        self.Rd = 448 if '' == label else int(label.split('_')[-1])
        # if 448 == self.Rd:  # skip this condition -> random
        #     self.register_buffer('sampled_indices', 
        #                          torch.LongTensor(range(self.Rd)))
        # else:
        #     self.register_buffer('sampled_indices',
        #                          torch.randperm(64 + 128 + 256)[:self.Rd])  # should be in saving state
        if self.preReLU:
            print('> SpadeResNet `preReLU` enabled')
            self.relu = nn.ReLU(inplace=False)
            for l in [self.module.layer1,
                      self.module.layer2,
                      self.module.layer3,
                      self.module.layer4]:
                l[-1] = MyBasicBlock(l[-1])

        if self.fast_dense:
            self.fastdense()

        self.layers = [
            self.module.layer1,
            self.module.layer2,
            self.module.layer3,
            self.module.layer4]

        self.evaluator = None

        # if self._debug:
            # print(self.sampled_indices)

    def fastdense(self):
        def transform_fastdense(m, namespace=[], visited=[]):
            pooling_count = 0
            for n, c in m.named_children():
                if (isinstance(c, nn.Conv2d) or isinstance(c, nn.MaxPool2d)) \
                    and hasattr(c, 'stride') and tuple_gt(c.stride, 1):
                    print('\t{} {} {} replaced with {}'.format(
                       ' '.join(namespace), n, type(c).__name__, 'Mul2d'))
                    setattr(m, n, Mul2d(c))
                    if 'downsample' not in namespace:
                        pooling_count += 1
                else:
                    if c not in visited and not isinstance(c, Mul2d):
                        pooling_count += transform_fastdense(
                            c, namespace + [n], visited + [c])
            return pooling_count
        for i in range(1, 4):
            transform_fastdense(self.layers[i])
        self.pooling_counts = [0, 1, 2, 3]
        self.fast_dense = True

    def reset_params(self):
        for m in self.module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def resample_dim(self):
        print('resample dimensions...')
        self.sampled_indices.zero_()
        self.sampled_indices += torch.randperm(64 + 128 + 256)[:self.Rd].to(
            self.sampled_indices.device)  # should be in saving state
        self.evaluator.init()

    def stem(self, x):
        x = self.module.conv1(x)
        x = self.module.bn1(x)
        x = self.module.relu(x)
        x = self.module.maxpool(x)
        return x

    def forward(self, x):
        sizes = x.size()[-2:]
        x = self.stem(x)
        outputs = []

        for i, b in enumerate(self.layers):
            if 0 < i and self.preReLU:
                x = self.relu(x)

            x = b(x)
            output = x
            if self.fast_dense:
                for j in range(self.pooling_counts[i]):
                    output = MulUnPool2d(2)(output)
            outputs.append(output)

        if self._debug:
            print(outputs[0].shape)  # b,  64, 56, 56
            print(outputs[1].shape)  # b, 128, 28, 28
            print(outputs[2].shape)  # b, 256, 14, 14

        if self.fast_dense:
            if 256 == self.Rd:
                outputs = [outputs[-2]]
            else:
                outputs = [torch.cat(outputs[:3], dim=1)]
        else:
            if 64 == self.Rd:
                outputs = [outputs[0]]
            elif 128 == self.Rd:
                outputs = [outputs[1]]
            elif 256 == self.Rd:
                outputs = [outputs[2]]
            else:
                outputs = [torch.cat([
                    outputs[0],
                    F.interpolate(outputs[1], scale_factor=2),
                    F.interpolate(outputs[2], scale_factor=4),
                    ], dim=1)]

        self.input_features = outputs
        if self.evaluator is None:
            return outputs[0]
        outputs = self.evaluator(outputs)
        output = 0
        for i, out in enumerate(outputs):
            if self.fast_dense:
                pass
            else:
                out = F.interpolate(out, scale_factor=2**i) if i > 0 else out
            output += out

        if not self.training:  # no_grad
            output = self.upsample(output, sizes)

        return [output]

    def remove_last_relu(self, layer):
        layer[-1] = MyBasicBlock(layer[-1], shortcut=False)

    def upsample(self, output, sizes):
        device = output.device
        if 3 == len(output.size()):
            output = output.unsqueeze(1)
        output = F.interpolate(output, size=sizes, mode='bilinear',
                               align_corners=False)
        output = np.stack([gaussian_filter(
            output[i].squeeze(0).cpu().detach().numpy(), sigma=4) for i in range(output.size(0))], axis=0)
        output = torch.Tensor(output).to(device).unsqueeze(1)
        return output


class SpadeMobilenetV3(nn.Module):
    def __init__(self, module, preReLU=False, label=''):
        super(SpadeMobilenetV3, self).__init__()
        self.module = module
        self.preReLU = preReLU or ('preReLU' in label)
        self._debug = False
        self.evaluator = None
        if self.preReLU:
            raise NotImplementedError()

    def forward(self, x):
        sizes = x.size()[-2:]
        indices = [3, 6, 12] if 16 == len(self.module.features) else [1, 3, 8]

        outputs = []
        for i in range(len(self.module.features)):
            x = self.module.features[i](x)
            if i in indices:
                outputs.append(x)

        if self._debug:
            print(outputs[0].shape)
            print(outputs[1].shape)
            print(outputs[2].shape)

        outputs = [torch.cat([
                outputs[0],
                F.interpolate(outputs[1], scale_factor=2),
                F.interpolate(outputs[2], scale_factor=4),
            ], dim=1)]

        self.input_features = outputs
        if self.evaluator is None:
            return outputs[0]
        outputs = self.evaluator(outputs)
        output = outputs[0]

        if not self.training:  # no_grad
            output = self.upsample(output, sizes)

        return [output]

    def upsample(self, output, sizes):
        device = output.device
        if 3 == len(output.size()):
            output = output.unsqueeze(1)
        output = F.interpolate(output, size=sizes, mode='bilinear',
                               align_corners=False)
        output = np.stack([gaussian_filter(
            output[i].squeeze(0).cpu().detach().numpy(), sigma=4) for i in range(output.size(0))], axis=0)
        output = torch.Tensor(output).to(device).unsqueeze(1)
        return output


class MahEvaluator(nn.Module):
    def __init__(self, cov, mean, emb, k=100, method='ortho', num_samples=None, 
                 eps=1e-2):
        super(MahEvaluator, self).__init__()
        self.cov = cov
        self.mean = mean
        self.eps = eps  # 1e-2 from Defard et al. (2021)
        self.method = method
        self.num_samples = num_samples
        self.k = k
        h, w, c, d = cov.size()
        P = self.build()  # hwmk
        self.register_buffer('P', P)
        self.register_buffer('map', emb)

    def forward(self, x):
        m = self.mean.transpose(2,1).transpose(1,0).unsqueeze(0)  # 1nhw
        if self.method in ['sample', 'ortho', 'gaussian']:
            x[0] = torch.einsum('nchw, ck -> nkhw', x[0], self.map)
        M = x[0] - m  # nchw
        if 'global' == self.method:
            R = torch.einsum('nmhw,mk,nkhw->nhw', M, self.P, M).unsqueeze(1)
        else:
            R = torch.einsum('nmhw,hwmk,nkhw->nhw', M, self.P, M).unsqueeze(1)
        return [R.abs().sqrt()]

    def build(self):
        print('build a precision matrix...')
        xx = self.cov

        if 'global' == self.method:
            xx = xx.mean(1).mean(0)
            I = torch.eye(xx.size(-1)).to(xx.device)
        else:
            I = torch.eye(xx.size(-1)).unsqueeze(0).unsqueeze(0).to(xx.device)

        if 'lowrank' == self.method:
            U, S, V = self.svd(xx + self.eps * I)
            D = S[...,:self.k].pow(-1)
            return torch.einsum('hwnm, hwm, hwkm -> hwnk', U[...,:self.k], D, V[...,:self.k])
        elif 'lowranki' == self.method:
            k = max(min(*self.cov.size()[:2]), self.k)
            U, S, V = self.svd(xx + self.eps * I)
            D = S[...,k-self.k:k].pow(-1)
            return torch.einsum('hwnm, hwm, hwkm -> hwnk', U[...,k-self.k:k], D, V[...,k-self.k:k])
        elif 'null' == self.method:  # including null vectors
            U, S, V = self.svd(xx + self.eps * I)
            D = S[...,-self.k:].pow(-1)
            return torch.einsum('hwnm, hwm, hwkm -> hwnk', U[...,-self.k:], D, V[...,-self.k:])
        else:
            return (xx + self.eps * I).inverse()

    def svd(self, x):
        f = 'cache/svd_{:d}_{:d}.pth'.format(self.num_samples, self.cov.size(-1))
        if os.path.isfile(f):
            U, S, V = torch.load(f)
        else:
            U, S, V = x.svd()
            torch.save((U, S, V), f)
        return U, S, V

    @staticmethod
    def get_embedding(fin, fout, method):
        W = torch.eye(fin)
        if 'sample' == method:
            s = torch.randperm(fin)[:fout]
            W = W[:, s]
        elif 'ortho' == method:
            W = torch.Tensor(fin, fout)
            nn.init.orthogonal_(W)
        elif 'gaussian' == method:
            W = torch.Tensor(fin, fout).normal_()
        return W


def compute_val_scores(val_loader, teacher_model, models, teacher_mean,
                       teacher_std, reduction='mean', device=None,
                       discriminator=None, args=None):
    e_score = None
    v_score = None
    b = val_loader.batch_size
    multiscale = 2 == len(teacher_mean.size()) or 4 == len(teacher_mean.size())
    net_device = list(models.parameters())[0].device
    for j, (x, y, a, c) in enumerate(tqdm(val_loader)):
        if e_score is None:
            if multiscale:
                e_score = torch.Tensor(len(val_loader.dataset), x.size(
                    2), x.size(3), teacher_mean.size(-1)).to(device)
                v_score = torch.Tensor(len(val_loader.dataset), x.size(
                    2), x.size(3), teacher_mean.size(-1)).to(device)
            else:
                e_score = torch.Tensor(
                    len(val_loader.dataset), x.size(2), x.size(3)).to(device)
                v_score = torch.Tensor(
                    len(val_loader.dataset), x.size(2), x.size(3)).to(device)
        with torch.no_grad():
            x = x.to(net_device)
            y_t = teacher_model(x)
            y_s = []
            for k in range(len(models)):
                try:
                    if models[k] is None:
                        y_s.append(module_list_clone(y_t, fill=0))
                    else:
                        y = models[k](x)
                        y_s.append(y)
                    use_stn = False
                except KeyError:
                    # TODO: ad-hoc
                    # exist named modules, e.g. fc_loc
                    use_stn = True

            t_mean, t_std = stn_stat(
                y_s, teacher_mean, teacher_std, models, use_stn)

            if multiscale:
                scores = get_multiscale_gaussian_modeling_score(
                    x, y_s, y_t, teacher_mean, teacher_std, args=args)
            else:
                e, v = get_gaussian_modeling_score(
                    x, y_s, y_t, teacher_mean, teacher_std, args=args)

        m = (x[:, 1] != 0).to(device)

        if multiscale:
            for k, score in enumerate(scores):
                e_score[j * b: j * b + x.size(0), :, :, k] = \
                    score[0].to(device) * m
                v_score[j * b: j * b + x.size(0), :, :, k] = \
                    score[1].to(device) * m
        else:
            e_score[j * b: j * b + x.size(0)] = e.to(device) * m
            v_score[j * b: j * b + x.size(0)] = v.to(device) * m

    if multiscale:
        if 'mean' == reduction:
            m = e_score[..., 0] != 0
            return [(
                e_score[..., i][m].mean(), e_score[..., i][m].std(),
                v_score[..., i][m].mean(), v_score[..., i][m].std()
            ) for i in range(e_score.size(-1))]
        else:
            return [(
                e_score[..., i].mean(0), e_score[..., i].std(0),
                v_score[..., i].mean(0), v_score[..., i].std(0)
            ) for i in range(e_score.size(-1))]
    else:
        if 'mean' == reduction:
            m = e_score[...] != 0
            return (e_score[m].mean(), e_score[m].std(),
                    v_score[m].mean(), v_score[m].std())
        else:
            return (e_score.mean(0), e_score.std(0),
                    v_score.mean(0), v_score.std(0))


if '__main__' == __name__:
    unittest.main()
