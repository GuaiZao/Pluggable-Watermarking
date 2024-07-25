import torch
import torch.nn as nn
from torchvision import transforms

class Blur(nn.Module):
    def __init__(self, degree):
        super().__init__()
        self.degree = degree

        self.blur = transforms.GaussianBlur(5,sigma=degree)
    def forward(self, x):
        source = x[1]
        x= self.blur(x[0])
        # noise = np.random.normal(0, self.degree, size=x.shape)
        # x = x + noise
        return x,source