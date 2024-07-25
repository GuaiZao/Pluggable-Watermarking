import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import cv2 as cv

# 实现四种攻击方法
# 输入一张图像
# 1. 裁剪 2. 模糊 3. jpeg压缩 4. 缩放
# 直接返回

# 裁剪 - 参数degree表示裁剪的占比
def createRandomCrop(size):
    tf_randam_crop = transforms.RandomCrop(size)
    return tf_randam_crop

class Crop(nn.Module):
    def __init__(self, degree):
        super().__init__()
        self.degree = degree

    def forward(self, x):
        w,h = x.size()[-1],x.size()[-2]
        size = (int(w*self.degree), int(h*self.degree))
        tf_randam_crop = createRandomCrop(size)
        x = tf_randam_crop(x)
        return x

# 模糊 - 参数degree表示噪声的标准差
class Blur(nn.Module):
    def __init__(self, degree):
        super().__init__()
        self.degree = degree

    def forward(self, x):
        noise = np.random.normal(0, self.degree, size=x.shape)
        x = x + noise
        return x

# 放缩 - 参数degree表示缩放系数
def createResize(size):
    tf_resize = transforms.Resize(size)
    return tf_resize

class ReSize(nn.Module):
    def __init__(self, degree):
        super().__init__()
        self.degree = degree
    def forward(self, x):
        w,h = x.size()[-1],x.size()[-2]
        size = (int(w*self.degree), int(h*self.degree))
        tf_resize = createResize(size)
        x = tf_resize(x)
        return x



# jpeg压缩 - 参数degree表示压缩系数
# 引入本地文件
from noise_layers.modules import compress_jpeg, decompress_jpeg
from noise_layers.utils_ import diff_round, quality_to_factor
import random
class DiffJPEG(nn.Module):
    def __init__(self, differentiable=True):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()

        # 生成随机quality
        quality_list = [60, 80, 90]
        quality_index = random.randint(0, len(quality_list)-1)
        quality = quality_list[quality_index]

        self.rounding = ''
        if differentiable:
            self.rounding = diff_round
        else:
            self.rounding = torch.round
        self.factor = quality_to_factor(quality)
        
    def forward(self, x):
        # print(x)
        source_image = x[1]
        x = x[0]
        height = x.size()[-2]
        width = x.size()[-1]
        compress = compress_jpeg(rounding=self.rounding, factor=self.factor).cuda()
        decompress = decompress_jpeg(height, width, rounding=self.rounding, factor=self.factor).cuda()

        x = x * 255. # PIL读取的图像，是0-1的，这里默认处理的是0-255的
        y, cb, cr = compress(x)
        recovered = decompress(y, cb, cr)
        recovered = recovered / 255.
        return recovered,source_image

    def set_quality(self, quality):
        factor = quality_to_factor(quality)
        self.compress.factor = factor
        self.decompress.factor = factor


    

