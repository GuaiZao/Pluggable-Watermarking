# Pytorch
import torch
import torch.nn as nn
# Local
from modules import compress_jpeg, decompress_jpeg
from utils import diff_round, quality_to_factor


class DiffJPEG(nn.Module):
    def __init__(self, differentiable=True, quality=80):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        self.rounding = ''
        if differentiable:
            self.rounding = diff_round
        else:
            self.rounding = torch.round
        self.factor = quality_to_factor(quality)
        
    def forward(self, x):
        height = x.shape[-2] 
        width = x.shape[-1]
        compress = compress_jpeg(rounding=self.rounding, factor=self.factor)
        decompress = decompress_jpeg(height, width, rounding=self.rounding, factor=self.factor)
        y, cb, cr = compress(x)
        recovered = decompress(y, cb, cr)
        return recovered

    def set_quality(self, quality):
        factor = quality_to_factor(quality)
        self.compress.factor = factor
        self.decompress.factor = factor


# if __name__ == '__main__':
#     with torch.no_grad():
#         import cv2
#         import numpy as np

#         img = cv2.imread("/home/lkm/lip_sync/FID/img_quality/approach/A_test_/05000_0.jpg")
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         print(img.shape)
#         inputs = np.transpose(img, (2, 0, 1))
#         inputs = inputs[np.newaxis, ...]
#         print(inputs.shape)
#         tensor = torch.FloatTensor(inputs)
#         jpeg = DiffJPEG(512, 512, differentiable=True)

#         quality = 80
#         jpeg.set_quality(quality)

#         outputs = jpeg(tensor)
#         outputs = outputs.detach().numpy()
#         print(outputs.shape)
#         outputs = np.transpose(outputs[0], (1, 2, 0))
#         print(outputs.shape)

#         outputs = cv2.cvtColor(outputs, cv2.COLOR_RGB2BGR)
        
#         cv2.imwrite("QF:"+str(quality)+'.jpg', outputs)
        # cv2.imshow("QF:"+str(quality), outputs / 255.)
        # cv2.waitKey()

        # from skimage.metrics import peak_signal_noise_ratio as PSNR
        # img = cv2.imread("Lena.png")
        # print(PSNR(np.uint8(outputs), np.uint8(img)))
