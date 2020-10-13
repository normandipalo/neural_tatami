import numpy as np
import matplotlib.pyplot as plt
import time

from torch.utils.data import Dataset, DataLoader
from PIL import Image

from hyper import cfg

class ImagesData(Dataset):
  def __init__(self, folder, y_path, fr, to, aug_fun_x = None, aug_fun_y = None):
    self.folder = folder
    self.mean = np.zeros((1,64,64))
    self.std = np.ones((1,64,64))
    self.y_mean = np.zeros((4))
    self.y_std = np.ones((4))
    self.fr, self.to = fr, to
    self.y = np.load(y_path)
    self.aug_fun_x = aug_fun_x
    self.aug_fun_y = aug_fun_y

  def __len__(self):
    return self.to - self.fr

  def __getitem__(self, i):
    i += self.fr
    a = Image.open(self.folder + "{}.png".format(i)).resize((64,64))
    a = np.asarray(a, dtype = np.float32)
    a = (a)/255
    y = self.y[i]
    if self.aug_fun_x:
        a = self.aug_fun_x(a)
    if self.aug_fun_y:
        y = self.aug_fun_y(y)

    return a.transpose((2,0,1)), y


class IRData(Dataset):
  def __init__(self, folder, y_path, fr, to, aug_fun_x = None, aug_fun_y = None):
    self.folder = folder
    self.mean = np.zeros((1,64,64))
    self.std = np.ones((1,64,64))
    self.y_mean = np.zeros((4))
    self.y_std = np.ones((4))
    self.fr, self.to = fr, to
    self.y = np.load(y_path)
    self.aug_fun_x = aug_fun_x
    self.aug_fun_y = aug_fun_y

  def __len__(self):
    return self.to - self.fr

  def __getitem__(self, i):
    i += self.fr
    a = Image.open(self.folder + "{}.png".format(i))
    a = np.asarray(a)
    a = (np.asarray(Image.fromarray(a).resize((64,64)), dtype = np.float32))
    y = self.y[i]
    if self.aug_fun_x:
        a = self.aug_fun_x(a)
    if self.aug_fun_y:
        y = self.aug_fun_y(y)

    return a[None,:,:]/255, y

class KPData(Dataset):
    def __init__(self, kps_path, y_path, fr, to, aug_fun_x = None, aug_fun_y = None):
        self.kps = np.load(kps_path)
        self.y = np.load(y_path)
        self.fr, self.to = fr, to
        self.aug_fun_x = aug_fun_x
        self.aug_fun_y = aug_fun_y

    def __len__(self):
        return self.to - self.fr

    def __getitem__(self, i):
        i += self.fr
        x, y = self.kps[i], self.y[i]
        if self.aug_fun_x:
            x = self.aug_fun(x)
        if self.aug_fun_y:
            y = self.aug_fun_y(y)
        return x, y


datas = {"RGB_data" : {"data_obj" : ImagesData, "len" : 150000, "in_dim" : 64, "in_chans" : 3},
        "IR_data" : {"data_obj" : IRData, "len" : 150000, "in_dim" : 64, "in_chans" : 1},
        "KP_data" : {"data_obj" : KPData, "len" : 150000, "in_dim" : 32, "in_chans" : 0}}
