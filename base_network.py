import timm
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import numpy as np
import torchvision
from torch.nn import init

from adain import ApplyStyle



# watermarking detector
class dis_xcep(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.xception_model = timm.create_model('xception', num_classes=dim, pretrained=True)
        self.result = {}
    def fix_model(self,l):
        for i,(name,m) in enumerate(self.xception_model.named_modules()):
            if i <l:
                m.requires_grad = False

    def forward(self,x):
        x = self.xception_model(x)
        return x
    

# for original torch.nn.Conv2D()
class get_mask_conv(nn.Module):
    def __init__(self,input_conv,gpu_ids):
        super().__init__()
        # self.input_conv = input_conv
        self.ori_weight = input_conv.weight.detach()
        shape = torch.ones((self.ori_weight.size(0),self.ori_weight.size(1),1,1)).cuda(gpu_ids)
        self.weight_mask = nn.Parameter(data=torch.ones_like(shape)*6,requires_grad=True).cuda(gpu_ids)

        if input_conv.bias is not None:
            self.ori_bias = input_conv.bias.detach()
            self.bias_mask = nn.parameter.Parameter(data=torch.ones_like(self.ori_bias)*6,requires_grad=True).cuda(gpu_ids)
        else:
            self.ori_bias = None
            self.bias_mask = None
        self.gpu_ids = gpu_ids
        self.sig = nn.ReLU6()
        self.padding = input_conv.padding
        self.stride = input_conv.stride
        self.dilation = input_conv.dilation
    def sign(self,x):
        return 0.5*(torch.sign(x)+1)

    def forward(self,x):
        res = list()
        for i in range(x.size(0)):
            new_weight = self.ori_weight * self.sig(self.weight_mask.repeat(1, 1, self.ori_weight.size(2), self.ori_weight.size(3))) / 6
            if self.ori_bias is not None:
                new_bias = self.ori_bias * self.sig(self.bias_mask)/6
                res.append(F.conv2d(x[i].unsqueeze(0), new_weight, new_bias, padding=self.padding,stride=self.stride,dilation=self.dilation).squeeze(0))
            else:
                res.append(F.conv2d(x[i].unsqueeze(0), new_weight, padding=self.padding, stride=self.stride,
                                    dilation=self.dilation).squeeze(0))

        middle_output = torch.stack(res, dim=0)
        return middle_output
    def get_weight_mask_loss(self,x):
        loss = self.sig(x)*self.sig(x)/36
        return loss.sum()/torch.numel(x)
    def get_rate(self):
        m = self.sig(self.weight_mask)/6
        return m.sum() / torch.numel(m)

# for original torch.nn.ConvTranspose2d()
class get_mask_conv_transpose(nn.Module):
    def __init__(self,input_conv,gpu_ids):
        super().__init__()
        # self.input_conv = input_conv
        self.ori_weight = input_conv.weight.detach()
        shape = torch.ones((self.ori_weight.size(0),self.ori_weight.size(1),1,1)).cuda(gpu_ids)
        self.weight_mask = nn.Parameter(data=torch.ones_like(shape)*6,requires_grad=True).cuda(gpu_ids)

        if input_conv.bias is not None:
            self.ori_bias = input_conv.bias.detach()
            self.bias_mask = nn.parameter.Parameter(data=torch.ones_like(self.ori_bias)*6,requires_grad=True).cuda(gpu_ids)
        else:
            self.ori_bias = None
            self.bias_mask = None
        self.gpu_ids = gpu_ids
        self.sig = nn.ReLU6()
        self.padding = input_conv.padding
        self.stride = input_conv.stride
        self.dilation = input_conv.dilation
        self.output_padding = input_conv.output_padding
    def sign(self,x):
        return 0.5*(torch.sign(x)+1)

    def forward(self,x):
        res = list()
        for i in range(x.size(0)):
            new_weight = self.ori_weight * self.sig(self.weight_mask.repeat(1, 1, self.ori_weight.size(2), self.ori_weight.size(3))) / 6
            if self.ori_bias is not None:
                new_bias = self.ori_bias * self.sig(self.bias_mask)/6
                res.append(F.conv_transpose2d(x[i].unsqueeze(0), new_weight, new_bias, padding=self.padding,output_padding=self.output_padding,stride=self.stride,dilation=self.dilation).squeeze(0))
            else:
                res.append(F.conv_transpose2d(x[i].unsqueeze(0), new_weight, padding=self.padding, output_padding=self.output_padding,stride=self.stride,
                                    dilation=self.dilation).squeeze(0))

        middle_output = torch.stack(res, dim=0)
        return middle_output
    def get_weight_mask_loss(self,x):
        loss = self.sig(x)*self.sig(x)/36
        return loss.sum()/torch.numel(x)
    def get_rate(self):
        m = self.sig(self.weight_mask)/6
        return m.sum() / torch.numel(m)


# for modulated conv
from math import sqrt
class get_mask_conv_modulated(nn.Module):
    def __init__(self,input_conv,gpu_ids):
        super().__init__()
        # self.input_conv = input_conv
        self.ori_weight = input_conv.weight.detach()
        shape = torch.ones((self.ori_weight.size(0),self.ori_weight.size(1),1,1)).cuda(gpu_ids)
        self.weight_mask = nn.Parameter(data=torch.ones_like(shape)*6,requires_grad=True).cuda(gpu_ids)

        if input_conv.bias is not None:
            self.ori_bias = input_conv.bias.detach()
            self.bias_mask = nn.parameter.Parameter(data=torch.ones_like(self.ori_bias)*6,requires_grad=True).cuda(gpu_ids)
        else:
            self.ori_bias = None
            self.bias_mask = None
        self.gpu_ids = gpu_ids
        self.sig = nn.ReLU6()

        self.in_channels = input_conv.in_channels
        self.out_channels = input_conv.out_channels
        self.kernel_size = input_conv.kernel_size
        self.mlp_class_std = input_conv.mlp_class_std
        self.padding = input_conv.padding
        self.blur = input_conv.blur
        self.padding = input_conv.padding
        self.upsample = input_conv.upsample
        if self.upsample:
            self.upsampler = input_conv.upsampler
        self.downsample = input_conv.downsample
        if self.downsample:
            self.down_sampler = input_conv.down_sampler
        self.demudulate = input_conv.demudulate

    def sign(self,x):
        return 0.5*(torch.sign(x)+1)


    def set_mask(self):
        self.f_w_weight = (6 - self.sig(self.w_mask.repeat(1, 1, self.ori_weight.size(2), self.ori_weight.size(3)))) / 6
        if self.b_mask is not None:
            self.f_w_bias = (6 - self.sig(self.b_mask)) / 6
    def forward(self,input,latent):

        fan_in = self.ori_weight.data.size(1) * self.ori_weight.data[0][0].numel()




        weight = self.ori_weight * sqrt(2 / fan_in)
        weight = weight.view(1, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        s = 1 + self.mlp_class_std(latent).view(-1, 1, self.in_channels, 1, 1)
        weight = s * weight
        if self.demudulate:
            d = torch.rsqrt((weight ** 2).sum(4).sum(3).sum(2) + 1e-5).view(-1, self.out_channels, 1, 1, 1)
            # weight = (d * weight).view(-1, self.in_channels, self.kernel_size, self.kernel_size)
            weight = d*weight


        if self.upsample:
            input = self.upsampler(input)

        if self.downsample:
            input = self.blur(input)

        new_weight = weight * self.sig(
            self.weight_mask.repeat(1, 1, self.ori_weight.size(2), self.ori_weight.size(3))).unsqueeze(0) / 6

        new_weight = new_weight.view(-1, self.in_channels, self.kernel_size, self.kernel_size)

        b,_,h,w = input.shape
        input = input.view(1,-1,h,w)
        input = self.padding(input)

        new_bias = self.ori_bias * self.sig(self.bias_mask) / 6
        # print(weight.size())
        out = F.conv2d(input, new_weight, groups=b).view(b, self.out_channels, h, w) + new_bias

        if self.downsample:
            out = self.downsampler(out)

        if self.upsample:
            out = self.blur(out)

        return out

    def get_weight_mask_loss(self,x):
        loss = self.sig(x)*self.sig(x)/36
        return loss.sum()/torch.numel(x)
    def get_rate(self):
        m = self.sig(self.weight_mask)/6
        return m.sum() / torch.numel(m)



class get_mask_conv_FusedUpsample(nn.Module):
    def __init__(self,input_conv,gpu_ids):
        super().__init__()

        self.ori_weight = input_conv.weight.detach()
        fan_in = self.ori_weight.size(0)*self.ori_weight.size(2)*self.ori_weight.size(3)
        self.multiplier = sqrt(2 / fan_in)
        self.ori_weight = F.pad(self.ori_weight * self.multiplier, [1, 1, 1, 1])
        self.ori_weight = (
                         self.ori_weight[:, :, 1:, 1:]
                         + self.ori_weight[:, :, :-1, 1:]
                         + self.ori_weight[:, :, 1:, :-1]
                         + self.ori_weight[:, :, :-1, :-1]
                 ) / 4
        shape = torch.ones((self.ori_weight.size(0),self.ori_weight.size(1),1,1)).cuda(gpu_ids)
        self.weight_mask = nn.Parameter(data=torch.ones_like(shape)*6,requires_grad=True).cuda(gpu_ids)

        if input_conv.bias is not None:
            self.ori_bias = input_conv.bias.detach()
            self.bias_mask = nn.parameter.Parameter(data=torch.ones_like(self.ori_bias)*6,requires_grad=True).cuda(gpu_ids)
        else:
            self.ori_bias = None
            self.bias_mask = None
        self.gpu_ids = gpu_ids
        self.sig = nn.ReLU6()

        self.pad = input_conv.pad

    def forward(self, input):
        new_weight = self.ori_weight * self.sig(self.weight_mask.repeat(1, 1, self.ori_weight.size(2), self.ori_weight.size(3))) / 6
        out = F.conv_transpose2d(input, new_weight, self.ori_bias, stride=2, padding=self.pad)

        return out
    def get_weight_mask_loss(self,x):
        loss = self.sig(x)*self.sig(x)/36
        return loss.sum()/torch.numel(x)
    def get_rate(self):
        m = self.sig(self.weight_mask)/6
        return m.sum() / torch.numel(m)



# for original torch.nn.Conv2D()
class get_fingerprint(nn.Module):
    def __init__(self, input_masked_conv, finger_dim):
        super().__init__()
        self.sig = nn.ReLU6()
        self.gpu_ids = input_masked_conv.gpu_ids
        self.finger_dim = finger_dim
        self.w_mask = input_masked_conv.weight_mask.detach()
        self.ori_weight = input_masked_conv.ori_weight.detach()

        self.sep = False
        if self.ori_weight.size(2) == 1:
            self.sep = True
            self.ori_weight_3 = torch.zeros(self.ori_weight.size(0),self.ori_weight.size(1),3,3).cuda()
            self.ori_weight_3[:, :, 1, 1] = self.ori_weight[:, :, 0, 0]
            self.ori_weight = self.ori_weight_3

        weight_shape = self.ori_weight.size()

        self.style_weight = ApplyStyle(128, weight_shape[1] * weight_shape[2] * weight_shape[3], fin_dim=finger_dim)
        self.f_weight = nn.parameter.Parameter(
            data=torch.ones(1, weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]).cuda(
                self.gpu_ids) * self.ori_weight, requires_grad=True).cuda(self.gpu_ids)
        self.f_w_weight = (6 - self.sig(self.w_mask.repeat(1, 1, self.ori_weight.size(2), self.ori_weight.size(3)))) / 6

        if input_masked_conv.bias_mask is None:
            self.b_mask = None
            self.ori_bias = None
            self.f_bias = None
        else:
            self.b_mask = input_masked_conv.bias_mask.detach()
            self.ori_bias = input_masked_conv.ori_bias.detach()
            bias_shape = input_masked_conv.ori_bias.size()
            self.style_bias = nn.Sequential(
                nn.Linear(in_features=finger_dim, out_features=bias_shape[0], bias=False),
                # nn.InstanceNorm1d(bias_shape[0]),
                nn.LeakyReLU(),
                nn.Linear(in_features=bias_shape[0], out_features=bias_shape[0], bias=False),
                nn.LeakyReLU()
            )
            self.f_bias = nn.parameter.Parameter(data=torch.ones(1, bias_shape[0]).cuda(self.gpu_ids) * self.ori_bias,
                                                 requires_grad=True).cuda(self.gpu_ids)
            self.f_w_bias = (6 - self.sig(self.b_mask)) / 6

        self.padding = input_masked_conv.padding
        if self.sep ==True:

            self.padding = (self.padding[0]+1,self.padding[1]+1)
        self.stride = input_masked_conv.stride
        self.dilation = input_masked_conv.dilation

        self.mean_ori_weight = self.ori_weight.mean()
        self.std_ori_weight = self.ori_weight.std()
        # self.mean_ori_bias = abs(self.ori_bias).mean()

    def set_mask(self):
        self.f_w_weight = (6 - self.sig(self.w_mask.repeat(1, 1, self.ori_weight.size(2), self.ori_weight.size(3)))) / 6
        if self.b_mask is not None:
            self.f_w_bias = (6 - self.sig(self.b_mask)) / 6

    def forward(self, x):

        y, finger_printor = x

        res = list()
        s_weight = self.style_weight(self.f_weight.repeat(finger_printor.size(0), 1, 1, 1, 1), finger_printor)
        if self.f_bias is not None:
            s_bias = self.f_bias.repeat(finger_printor.size(0), 1) * self.style_bias(finger_printor)
            for i in range(y.size(0)):
                new_weight = (1 - self.f_w_weight) * self.ori_weight + self.f_w_weight * s_weight[i]
                std_weight = torch.std(new_weight)
                mean_weight = torch.mean(new_weight)
                new_weight = ((new_weight - mean_weight) / std_weight * self.std_ori_weight + self.mean_ori_weight)
                new_weight = (1 - self.f_w_weight) * self.ori_weight + self.f_w_weight * new_weight
                new_bias = (1 - self.f_w_bias) * self.ori_bias + self.f_w_bias * s_bias[i]
                res.append(F.conv2d(y[i].unsqueeze(0), new_weight, new_bias, padding=self.padding, stride=self.stride,
                                    dilation=self.dilation).squeeze(0))

        else:
            for i in range(y.size(0)):
                new_weight = (1 - self.f_w_weight) * self.ori_weight + self.f_w_weight * s_weight[i]
                std_weight = torch.std(new_weight)
                mean_weight = torch.mean(new_weight)
                new_weight = ((new_weight - mean_weight) / std_weight * self.std_ori_weight + self.mean_ori_weight)
                new_weight = (1 - self.f_w_weight) * self.ori_weight + self.f_w_weight * new_weight
                res.append(F.conv2d(y[i].unsqueeze(0), new_weight, padding=self.padding, stride=self.stride,
                                    dilation=self.dilation).squeeze(0))

        middle_output = torch.stack(res, dim=0)
        return middle_output

    def get_weight_mask_loss(self, x):
        loss = self.sig(x) * self.sig(x)
        return loss.sum() / torch.numel(x)

    def hook_fn(self, module, input, fingerprint):
        input_tuple = tuple(input, fingerprint)
        return input_tuple

    def get_fin_weight(self):
        return (6 - self.sig(self.w_mask.repeat(1, 1, self.ori_weight.size(2), self.ori_weight.size(3)))) / 6

    def get_rate(self):
        m = self.get_fin_weight()
        return m.sum() / torch.numel(m)

    def get_masked_weight(self):
        return (1 - self.f_w_weight) * self.ori_weight

    def get_fingered_weight(self):
        finger = torch.randint(0, 2, (1, self.finger_dim), dtype=torch.float).cuda()
        s_weight = self.style_weight(self.f_weight, finger)
        new_weight = (1 - self.f_w_weight) * self.ori_weight + self.f_w_weight * s_weight[0]
        std_weight = torch.std(new_weight)
        mean_weight = torch.mean(new_weight)
        new_weight = ((new_weight - mean_weight) / std_weight * self.std_ori_weight + self.mean_ori_weight)
        new_weight = (1 - self.f_w_weight) * self.ori_weight + self.f_w_weight * new_weight
        return new_weight


class get_fingerprint_FuseUpsample(nn.Module):
    def __init__(self, input_masked_conv, finger_dim):
        super().__init__()
        self.sig = nn.ReLU6()
        self.gpu_ids = input_masked_conv.gpu_ids
        self.finger_dim = finger_dim
        self.w_mask = input_masked_conv.weight_mask.detach()
        self.ori_weight = input_masked_conv.ori_weight.detach()
        weight_shape = input_masked_conv.ori_weight.size()
        self.style_weight = ApplyStyle(128, weight_shape[0] * weight_shape[2] * weight_shape[3], fin_dim=finger_dim,transpose=True)
        self.f_weight = nn.parameter.Parameter(
            data=torch.ones(1, weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]).cuda(
                self.gpu_ids) * self.ori_weight, requires_grad=True).cuda(self.gpu_ids)
        self.f_w_weight = (6 - self.sig(self.w_mask.repeat(1, 1, self.ori_weight.size(2), self.ori_weight.size(3)))) / 6

        if input_masked_conv.bias_mask is None:
            self.b_mask = None
            self.ori_bias = None
            self.f_bias = None
        else:
            self.b_mask = input_masked_conv.bias_mask.detach()
            self.ori_bias = input_masked_conv.ori_bias.detach()
            bias_shape = input_masked_conv.ori_bias.size()
            self.style_bias = nn.Sequential(
                nn.Linear(in_features=finger_dim, out_features=bias_shape[0], bias=False),
                nn.LeakyReLU(),
                nn.Linear(in_features=bias_shape[0], out_features=bias_shape[0], bias=False),
                nn.LeakyReLU()
            )
            self.f_bias = nn.parameter.Parameter(data=torch.ones(1, bias_shape[0]).cuda(self.gpu_ids) * self.ori_bias,
                                                 requires_grad=True).cuda(self.gpu_ids)
            self.f_w_bias = (6 - self.sig(self.b_mask)) / 6

        self.pad = input_masked_conv.pad


        self.mean_ori_weight = self.ori_weight.mean()
        self.std_ori_weight = self.ori_weight.std()


    def set_mask(self):
        self.f_w_weight = (6 - self.sig(self.w_mask.repeat(1, 1, self.ori_weight.size(2), self.ori_weight.size(3)))) / 6
        if self.b_mask is not None:
            self.f_w_bias = (6 - self.sig(self.b_mask)) / 6

    def forward(self, x):

        y, finger_printor = x
        res = list()
        s_weight = self.style_weight(self.f_weight.repeat(finger_printor.size(0), 1, 1, 1, 1), finger_printor)

        if self.f_bias is not None:
            s_bias = self.f_bias.repeat(finger_printor.size(0), 1) * self.style_bias(finger_printor)
            for i in range(y.size(0)):
                new_weight = (1 - self.f_w_weight) * self.ori_weight + self.f_w_weight * s_weight[i]
                std_weight = torch.std(new_weight)
                mean_weight = torch.mean(new_weight)
                new_weight = ((new_weight - mean_weight) / std_weight * self.std_ori_weight + self.mean_ori_weight)
                new_weight = (1 - self.f_w_weight) * self.ori_weight + self.f_w_weight * new_weight
                new_bias = (1 - self.f_w_bias) * self.ori_bias + self.f_w_bias * s_bias[i]
                res.append(F.conv_transpose2d(y[i].unsqueeze(0), new_weight, new_bias, stride=2,padding=self.pad).squeeze(0))

        else:
            for i in range(y.size(0)):
                new_weight = (1 - self.f_w_weight) * self.ori_weight + self.f_w_weight * s_weight[i]
                std_weight = torch.std(new_weight)
                mean_weight = torch.mean(new_weight)
                new_weight = ((new_weight - mean_weight) / std_weight * self.std_ori_weight + self.mean_ori_weight)
                new_weight = (1 - self.f_w_weight) * self.ori_weight + self.f_w_weight * new_weight
                res.append(F.conv_transpose2d(y[i].unsqueeze(0), new_weight, stride=2,padding=self.pad).squeeze(0))

        middle_output = torch.stack(res, dim=0)
        return middle_output

    def get_weight_mask_loss(self, x):
        loss = self.sig(x) * self.sig(x)
        return loss.sum() / torch.numel(x)

    def hook_fn(self, module, input, fingerprint):
        input_tuple = tuple(input, fingerprint)
        return input_tuple

    def get_fin_weight(self):
        return (6 - self.sig(self.w_mask.repeat(1, 1, self.ori_weight.size(2), self.ori_weight.size(3)))) / 6

    def get_rate(self):
        m = self.get_fin_weight()
        return m.sum() / torch.numel(m)

    def get_masked_weight(self):
        return (1 - self.f_w_weight) * self.ori_weight


class get_fingerprint_modulated(nn.Module):
    def __init__(self, input_masked_conv, finger_dim):
        super().__init__()
        self.sig = nn.ReLU6()
        self.gpu_ids = input_masked_conv.gpu_ids

        self.w_mask = input_masked_conv.weight_mask.detach()
        self.ori_weight = input_masked_conv.ori_weight.detach()
        weight_shape = input_masked_conv.ori_weight.size()
        self.style_weight = ApplyStyle(128, weight_shape[1] * weight_shape[2] * weight_shape[3], fin_dim=finger_dim)
        self.f_weight = nn.parameter.Parameter(
            data=torch.ones(1, weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]).cuda(
                self.gpu_ids) * self.ori_weight, requires_grad=True).cuda(self.gpu_ids)
        self.f_w_weight = (6 - self.sig(self.w_mask.repeat(1, 1, self.ori_weight.size(2), self.ori_weight.size(3)))) / 6

        if input_masked_conv.bias_mask is None:
            self.b_mask = None
            self.ori_bias = None
            self.f_bias = None
        else:
            self.b_mask = input_masked_conv.bias_mask.detach()
            self.ori_bias = input_masked_conv.ori_bias.detach()
            bias_shape = input_masked_conv.ori_bias.size()
            self.style_bias = nn.Sequential(
                nn.Linear(in_features=finger_dim, out_features=bias_shape[1], bias=False),
                # nn.InstanceNorm1d(bias_shape[0]),
                nn.LeakyReLU(),
                nn.Linear(in_features=bias_shape[1], out_features=bias_shape[1], bias=False),
                nn.LeakyReLU()
            )
            self.f_bias = nn.parameter.Parameter(data=torch.ones(1, bias_shape[1],1,1).cuda(self.gpu_ids) * self.ori_bias,
                                                 requires_grad=True).cuda(self.gpu_ids)
            self.f_w_bias = (6 - self.sig(self.b_mask)) / 6



        self.in_channels = input_masked_conv.in_channels
        self.out_channels = input_masked_conv.out_channels
        self.kernel_size = input_masked_conv.kernel_size
        self.mlp_class_std = input_masked_conv.mlp_class_std
        self.padding = input_masked_conv.padding
        self.blur = input_masked_conv.blur
        self.padding = input_masked_conv.padding
        self.upsample = input_masked_conv.upsample
        if self.upsample:
            self.upsampler = input_masked_conv.upsampler
        self.downsample = input_masked_conv.downsample
        if self.downsample:
            self.down_sampler = input_masked_conv.down_sampler
        self.demudulate = input_masked_conv.demudulate



    def forward(self, x):

        input,latent, finger_printor = x



        fan_in = self.ori_weight.data.size(1) * self.ori_weight.data[0][0].numel()

        weight = self.ori_weight * sqrt(2 / fan_in)
        weight = weight.view(1, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.mean_ori_weight = weight.mean()
        self.std_ori_weight = weight.std()

        s = 1 + self.mlp_class_std(latent).view(-1, 1, self.in_channels, 1, 1)
        weight = s * weight

        if self.demudulate:
            d = torch.rsqrt((weight ** 2).sum(4).sum(3).sum(2) + 1e-5).view(-1, self.out_channels, 1, 1, 1)
            weight = d*weight


        new_bias = self.ori_bias * self.sig(self.b_mask) / 6


        s_weight = self.style_weight(self.f_weight.repeat(finger_printor.size(0), 1, 1, 1, 1), finger_printor)
        s_bias = self.f_w_bias.repeat(finger_printor.size(0),1,1,1) * self.style_bias(finger_printor).view(finger_printor.size(0),-1,1,1)
        # s_bias = self.f_bias.repeat(finger_printor.size(0), 1) * self.style_bias(finger_printor)

        new_weight = (1 - self.f_w_weight.unsqueeze(0)) * weight + self.f_w_weight.unsqueeze(0) * s_weight

        std_weight = torch.std(new_weight,dim=(1,2,3,4))
        mean_weight = torch.mean(new_weight,dim=(1,2,3,4))
        # print(mean_weight.size())
        new_weight = ((new_weight - mean_weight.view(-1,1,1,1,1)) / std_weight.view(-1,1,1,1,1) * self.std_ori_weight + self.mean_ori_weight)
        new_weight = (1 - self.f_w_weight.unsqueeze(0)) * weight + self.f_w_weight.unsqueeze(0) * new_weight
        new_weight = new_weight.view(-1, self.in_channels, self.kernel_size, self.kernel_size)

        if self.upsample:
            input = self.upsampler(input)

        if self.downsample:
            input = self.blur(input)

        b,_,h,w = input.shape
        input = input.view(1,-1,h,w)
        input = self.padding(input)

        out = F.conv2d(input, new_weight, groups=b).view(b, self.out_channels, h, w) + new_bias
        out = out + s_bias

        if self.downsample:
            out = self.downsampler(out)

        if self.upsample:
            out = self.blur(out)

        return out


    def hook_fn(self, module, input, latent,fingerprint):
        input_tuple = tuple(input, latent, fingerprint)
        return input_tuple

    def get_fin_weight(self):
        return (6 - self.sig(self.w_mask.repeat(1, 1, self.ori_weight.size(2), self.ori_weight.size(3)))) / 6
    def get_rate(self):
        m = self.get_fin_weight()
        return m.sum() / torch.numel(m)
    def set_mask(self):
        self.f_w_weight = (6 - self.sig(self.w_mask.repeat(1, 1, self.ori_weight.size(2), self.ori_weight.size(3)))) / 6
        if self.b_mask is not None:
            self.f_w_bias = (6 - self.sig(self.b_mask)) / 6
    def get_masked_weight(self):
        return (1-self.f_w_weight) * self.ori_weight
    def get_fingered_weight(self):
        return (1 - self.f_w_weight) * self.ori_weight + self.f_w_weight * self.f_weight[0]



import torchvision.models as models
class VGG19Loss(nn.Module):
    """
    Perceptual loss based on VGG19
    """

    def __init__(self, device):
        super(VGG19Loss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features.to(device)
        self.layers = {
            '3': "relu1_2",
            '8': "relu2_2",
            '17': "relu3_3",
            '26': "relu4_3"
        }

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_features = self.get_features(input)
        target_features = self.get_features(target)

        loss = 0.0
        for layer in self.layers:
            loss += F.mse_loss(input_features[self.layers[layer]], target_features[self.layers[layer]])
        return loss

    def get_features(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features