import torch.nn as nn


class ResnetBlock_Adain(nn.Module):
    def __init__(self, dim, latent_size, padding_type, activation=nn.LeakyReLU(0.3,True)):
        super(ResnetBlock_Adain, self).__init__()

        p = 0
        conv1 = []
        if padding_type == 'reflect':
            conv1 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv1 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv1 += [nn.Conv2d(dim, dim, kernel_size=3, padding=p,bias=True), InstanceNorm()]
        self.conv1 = nn.Sequential(*conv1)
        self.style1 = ApplyStyle(latent_size, dim)
        self.act1 = activation

        p = 0
        conv2 = []
        if padding_type == 'reflect':
            conv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv2 += [nn.Conv2d(dim, dim, kernel_size=3, padding=p,bias=True), InstanceNorm()]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = ApplyStyle(latent_size, dim)
        self.norm=nn.InstanceNorm2d(dim)
    def forward(self, x, dlatents_in_slice):
        y = self.conv1(x)
        y = self.style1(y, dlatents_in_slice)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.style2(y, dlatents_in_slice)
        out = x + y
        out=self.norm(out)
        return out

class ApplyStyle(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """

    def __init__(self, latent_size, channels,fin_dim,transpose=False):

        super(ApplyStyle, self).__init__()
        self.channels = channels
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=fin_dim,out_features=fin_dim,bias=False),
            nn.InstanceNorm1d(fin_dim),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=fin_dim,out_features=self.channels,bias=False),
            nn.InstanceNorm1d(self.channels),
            nn.LeakyReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels,out_features=channels*2,bias=False),
            # nn.InstanceNorm1d(channels*2),
            # nn.LeakyReLU()
            # nn.ReLU()
        )
        self.transpose = transpose
    def forward(self, x, latent):
        # print(latent)
        if self.transpose:
            x = x.transpose(1, 2)
            # print(x.size())
        kernel_size = x.size(3)
        x = x.contiguous().view(x.size(0),x.size(1),-1)
        style = self.fc1(latent)
        style = self.fc2(style)
        style = self.linear(style)
        # print(style.size())
        # style => [batch_size, n_channels*2]
        shape = [-1, 2, 1, x.size(2)]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        # x = x * (style[:, 0] + 1.) + style[:, 1]
        x = x * (style[:, 0] * 1.+1) + style[:, 1] * 1
        x = x.view(x.size(0),x.size(1),-1,kernel_size,kernel_size)
        if self.transpose:
            x = x.transpose(1,2)
        return x


class generate_block(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """

    def __init__(self, latent_size, channels,fin_dim):

        super(generate_block, self).__init__()
        self.channels = channels
        self.sum_channels = channels[0]*channels[1]*channels[2]*channels[3]
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=fin_dim,out_features=fin_dim,bias=False),
            nn.InstanceNorm1d(fin_dim),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=fin_dim,out_features=self.sum_channels,bias=False),
            # nn.InstanceNorm1d(self.sum_channels),
            # nn.LeakyReLU()
        )
        # self.linear = nn.Sequential(
        #     nn.Linear(in_features=self.sum_channels,out_features=self.sum_channels,bias=False),
        #     # nn.InstanceNorm1d(channels*2),
        #     # nn.LeakyReLU()
        #     # nn.ReLU()
        # )
    def forward(self,latent):
        # pri
        style = self.fc1(latent)
        style = self.fc2(style)
        # style = self.linear(style)
        # print(style)
        # style => [batch_size, n_channels*2]
        style = style.view(style.size(0),self.channels[0],self.channels[1],self.channels[2],self.channels[3])
        return style

class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x)  # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp