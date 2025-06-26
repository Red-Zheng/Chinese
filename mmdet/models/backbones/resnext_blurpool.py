# Copyright (c) OpenMMLab. All rights reserved.
import math
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmdet.registry import MODELS
from ..layers import ResLayer
# from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.model import BaseModule
import torch.utils.checkpoint as cp
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from ..layers import SELayer

class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)),
                          int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels
 
        if self.filt_size == 1:
            a = np.array([1., ])
        elif self.filt_size == 2:
            a = np.array([1., 1.])
        elif self.filt_size == 3:
            a = np.array([1., 2., 1.])
        elif self.filt_size == 4:
            a = np.array([1., 3., 3., 1.])
        elif self.filt_size == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif self.filt_size == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif self.filt_size == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
 
        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
 
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)
 
    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1]) 
 
def get_pad_layer(pad_type):
    if pad_type in ['refl', 'reflect']:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ['repl', 'replicate']:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == 'zero':
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer

class BasicBlock(BaseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

class _Bottleneck(BaseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(_Bottleneck, self).__init__(init_cfg)
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        self.BlurPool = BlurPool(planes * self.expansion, stride=1, pad_off=0)

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(out)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            # out = self.eca(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)
                
            # out = self.eca(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)
            
            # print(out.shape)

            out = self.BlurPool(out)
            # out = self.eca(out)
            # out = self.se(out)

            
            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

class Bottleneck(_Bottleneck):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 groups=1,
                 base_width=4,
                 base_channels=64,
                 **kwargs):
        """Bottleneck block for ResNeXt.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(inplanes, planes, **kwargs)

        if groups == 1:
            width = self.planes
        else:
            width = math.floor(self.planes *
                               (base_width / base_channels)) * groups

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, width, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            self.norm_cfg, width, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            self.norm_cfg, self.planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.inplanes,
            width,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = self.dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                self.conv_cfg,
                width,
                width,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation,
                groups=groups,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                self.dcn,
                width,
                width,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation,
                groups=groups,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            self.conv_cfg,
            width,
            self.planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        if self.with_plugins:
            self._del_block_plugins(self.after_conv1_plugin_names +
                                    self.after_conv2_plugin_names +
                                    self.after_conv3_plugin_names)
            self.after_conv1_plugin_names = self.make_block_plugins(
                width, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                width, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                self.planes * self.expansion, self.after_conv3_plugins)

    def _del_block_plugins(self, plugin_names):
        """delete plugins for block if exist.

        Args:
            plugin_names (list[str]): List of plugins name to delete.
        """
        assert isinstance(plugin_names, list)
        for plugin_name in plugin_names:
            del self._modules[plugin_name]

@MODELS.register_module()
class ResNeXt_blurpool(ResNet):
    """ResNeXt backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        groups (int): Group of resnext.
        base_width (int): Base width of resnext.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, groups=1, base_width=4, **kwargs):
        self.groups = groups
        self.base_width = base_width
        super(ResNeXt_blurpool, self).__init__(**kwargs)

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``"""
        return ResLayer(
            groups=self.groups,
            base_width=self.base_width,
            base_channels=self.base_channels,
            **kwargs)
            
# class _Bottleneck(BaseModule):
#     expansion = 4

#     def __init__(self,
#                  inplanes,
#                  planes,
#                  stride=1,
#                  dilation=1,
#                  downsample=None,
#                  style='pytorch',
#                  with_cp=False,
#                  conv_cfg=None,
#                  norm_cfg=dict(type='BN'),
#                  dcn=None,
#                  plugins=None,
#                  init_cfg=None):
#         """Bottleneck block for ResNet.

#         If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
#         it is "caffe", the stride-two layer is the first 1x1 conv layer.
#         """
#         super(_Bottleneck, self).__init__(init_cfg)
#         assert style in ['pytorch', 'caffe']
#         assert dcn is None or isinstance(dcn, dict)
#         assert plugins is None or isinstance(plugins, list)
#         if plugins is not None:
#             allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
#             assert all(p['position'] in allowed_position for p in plugins)

#         self.inplanes = inplanes
#         self.planes = planes
#         self.stride = stride
#         self.dilation = dilation
#         self.style = style
#         self.with_cp = with_cp
#         self.conv_cfg = conv_cfg
#         self.norm_cfg = norm_cfg
#         self.dcn = dcn
#         self.with_dcn = dcn is not None
#         self.plugins = plugins
#         self.with_plugins = plugins is not None

#         if self.with_plugins:
#             # collect plugins for conv1/conv2/conv3
#             self.after_conv1_plugins = [
#                 plugin['cfg'] for plugin in plugins
#                 if plugin['position'] == 'after_conv1'
#             ]
#             self.after_conv2_plugins = [
#                 plugin['cfg'] for plugin in plugins
#                 if plugin['position'] == 'after_conv2'
#             ]
#             self.after_conv3_plugins = [
#                 plugin['cfg'] for plugin in plugins
#                 if plugin['position'] == 'after_conv3'
#             ]

#         if self.style == 'pytorch':
#             self.conv1_stride = 1
#             self.conv2_stride = stride
#         else:
#             self.conv1_stride = stride
#             self.conv2_stride = 1

#         self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
#         self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
#         self.norm3_name, norm3 = build_norm_layer(
#             norm_cfg, planes * self.expansion, postfix=3)

#         self.conv1 = build_conv_layer(
#             conv_cfg,
#             inplanes,
#             planes,
#             kernel_size=1,
#             stride=self.conv1_stride,
#             bias=False)
#         self.add_module(self.norm1_name, norm1)
#         fallback_on_stride = False
#         if self.with_dcn:
#             fallback_on_stride = dcn.pop('fallback_on_stride', False)
#         if not self.with_dcn or fallback_on_stride:
#             self.conv2 = build_conv_layer(
#                 conv_cfg,
#                 planes,
#                 planes,
#                 kernel_size=3,
#                 stride=self.conv2_stride,
#                 padding=dilation,
#                 dilation=dilation,
#                 bias=False)
#         else:
#             assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
#             self.conv2 = build_conv_layer(
#                 dcn,
#                 planes,
#                 planes,
#                 kernel_size=3,
#                 stride=self.conv2_stride,
#                 padding=dilation,
#                 dilation=dilation,
#                 bias=False)

#         self.add_module(self.norm2_name, norm2)
#         self.conv3 = build_conv_layer(
#             conv_cfg,
#             planes,
#             planes * self.expansion,
#             kernel_size=1,
#             bias=False)
#         self.add_module(self.norm3_name, norm3)

#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample

#         if self.with_plugins:
#             self.after_conv1_plugin_names = self.make_block_plugins(
#                 planes, self.after_conv1_plugins)
#             self.after_conv2_plugin_names = self.make_block_plugins(
#                 planes, self.after_conv2_plugins)
#             self.after_conv3_plugin_names = self.make_block_plugins(
#                 planes * self.expansion, self.after_conv3_plugins)

#     def make_block_plugins(self, in_channels, plugins):
#         """make plugins for block.

#         Args:
#             in_channels (int): Input channels of plugin.
#             plugins (list[dict]): List of plugins cfg to build.

#         Returns:
#             list[str]: List of the names of plugin.
#         """
#         assert isinstance(plugins, list)
#         plugin_names = []
#         for plugin in plugins:
#             plugin = plugin.copy()
#             name, layer = build_plugin_layer(
#                 plugin,
#                 in_channels=in_channels,
#                 postfix=plugin.pop('postfix', ''))
#             assert not hasattr(self, name), f'duplicate plugin {name}'
#             self.add_module(name, layer)
#             plugin_names.append(name)
#         return plugin_names

#     def forward_plugin(self, x, plugin_names):
#         out = x
#         for name in plugin_names:
#             out = getattr(self, name)(out)
#         return out

#     @property
#     def norm1(self):
#         """nn.Module: normalization layer after the first convolution layer"""
#         return getattr(self, self.norm1_name)

#     @property
#     def norm2(self):
#         """nn.Module: normalization layer after the second convolution layer"""
#         return getattr(self, self.norm2_name)

#     @property
#     def norm3(self):
#         """nn.Module: normalization layer after the third convolution layer"""
#         return getattr(self, self.norm3_name)

#     def forward(self, x):
#         """Forward function."""

#         def _inner_forward(x):
#             identity = x
#             out = self.conv1(x)
#             out = self.norm1(out)
#             out = self.relu(out)

#             if self.with_plugins:
#                 out = self.forward_plugin(out, self.after_conv1_plugin_names)

#             out = self.conv2(out)
#             out = self.norm2(out)
#             out = self.relu(out)

#             if self.with_plugins:
#                 out = self.forward_plugin(out, self.after_conv2_plugin_names)

#             out = self.conv3(out)
#             out = self.norm3(out)

#             if self.with_plugins:
#                 out = self.forward_plugin(out, self.after_conv3_plugin_names)

#             if self.downsample is not None:
#                 identity = self.downsample(x)

#             out += identity

#             return out

#         if self.with_cp and x.requires_grad:
#             out = cp.checkpoint(_inner_forward, x)
#         else:
#             out = _inner_forward(x)

#         out = self.relu(out)

#         return out


