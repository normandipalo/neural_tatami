import torch
import torch.nn as nn

from hyper import cfg


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class CNNet(nn.Module):
  def __init__(self, in_dim = 32, in_chans = 3, filters = [32, 32, 64, 128, 1], kernels = [5, 3, 3, 3, 1], dim_s = [1,1,2,2,1], activation = "selu", drop_p = 0.1):
    super().__init__()
    self.convs = nn.ModuleList([Conv2dAuto(in_channels = in_chans, out_channels = filters[0], stride = dim_s[0], kernel_size = kernels[0])] +
                               [Conv2dAuto(in_channels = filters[i-1], out_channels = filters[i], stride = dim_s[i], kernel_size = kernels[i]) for i in range(1, len(filters))])
    self.batch_norms = nn.ModuleList([nn.BatchNorm2d(num_features= filters[i]) for i in range(len(filters))])
    self.setup(torch.rand(1,in_chans, in_dim, in_dim))
    self.denses = nn.ModuleList([nn.Linear(in_features = self.lin_in_shape, out_features=128),
                                 nn.Linear(in_features = 128, out_features=4)])
    self.act = activation_func(activation)
    self.drop = nn.Dropout(p=drop_p)

  def setup(self, x):
    for l in self.convs:
      x = l(x)
    self.lin_in_shape = x.shape[-1]*x.shape[-2]*x.shape[-3]

  def forward(self, x):
    for l, b in zip(self.convs, self.batch_norms):
      x = self.drop(self.act(b(l(x))))
    x = x.view(-1, self.lin_in_shape)
    x = self.denses[0](x)
    x = self.act(x)
    x = self.drop(x)
    x = self.denses[1](x)
    return x

nets = {"cnn" : CNNet}
