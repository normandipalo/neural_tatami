import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch_optimizer as optim

from hyper import cfg, model_hp
from models import *
from data import *

writer = SummaryWriter()


net_func = nets[cfg["model"]]

data_obj, data_len = datas[cfg["data"]]["data_obj"], datas[cfg["data"]]["len"]
train_from, train_to = 0, int(data_len*(1 - cfg["validation_split"]))

d_train = data_obj(cfg["data_folder"], cfg["y_path"], train_from, train_to)
d_val = data_obj(cfg["data_folder"], cfg["y_path"], train_to, data_len)

dl_train = DataLoader(d_train, batch_size=cfg["batch_size"],
                        shuffle=True, num_workers=1)

dl_val = DataLoader(d_val, batch_size=cfg["batch_size"],
                        shuffle=False, num_workers=1)

n = net_func(in_dim = datas[cfg["data"]]["in_dim"], in_chans = datas[cfg["data"]]["in_chans"], **model_hp)

model_parameters = filter(lambda p: p.requires_grad, n.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print("Model parameters:", params)

opt = cfg["optimizer"](n.parameters(), lr = 1e-3)

device = cfg["device"]
n.double().to(device)

j = 0
for epoch in range(cfg["epochs"]):
    n.train()
    start = time.time()
    for i, (X, y_t) in enumerate(dl_train):
        opt.zero_grad()
        y_out = n(X.double().to(device))
        y_t = y_t.to(device)
        tot_loss = None
        for l in cfg["losses"]:
            y_l = l["pre_proc"](y_out)
            y_t_l = l["pre_proc"](y_t)
            loss = l["l_func"](y_l, y_t_l)*l["weight"]
            writer.add_scalar("Loss/{}".format(l["name"]), loss, epoch*j + i)
            if not tot_loss: tot_loss=loss
            else: tot_loss += loss
        tot_loss.backward()
        opt.step()
    j = i
    print("Steps:", j)
    print("Time for an epoch:", time.time() - start)
