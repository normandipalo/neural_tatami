import numpy as np
import matplotlib.pyplot as plt
import time
import hashlib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_sch

import torch_optimizer as optim

from hyper import cfg, cos_loss
from models import *
from data import *
from utils import *


model_hp = cfg["model_hp"]
net_func = nets[cfg["model"]]

data_obj, data_len = datas[cfg["data"]]["data_obj"], datas[cfg["data"]]["len"]
train_from, train_to = 0, int(data_len*(1 - cfg["validation_split"]))

d_train = data_obj(cfg["data_folder"], cfg["y_path"], train_from, train_to)
d_val = data_obj(cfg["data_folder"], cfg["y_path"], train_to, data_len)

dl_train = DataLoader(d_train, batch_size=cfg["batch_size"],
                        shuffle=True, num_workers=2)

dl_val = DataLoader(d_val, batch_size=cfg["batch_size"],
                        shuffle=False, num_workers=2)

n = net_func(in_dim = datas[cfg["data"]]["in_dim"], in_chans = datas[cfg["data"]]["in_chans"], **model_hp)

model_parameters = filter(lambda p: p.requires_grad, n.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print("Model parameters:", params)

opt = cfg["optimizer"](n.parameters(), lr = cfg["lr"])
scheduler = lr_sch.StepLR(opt, step_size=cfg["sch_step"], gamma=cfg["sch_gamma"])

device = cfg["device"]
n.float().to(device)

hash = store_hp(cfg)
writer = SummaryWriter("runs/pars_" + str(params) + "_model_" + cfg["model"] + "_data_" + cfg["data"] + "_" + str(hash))

j = 0
start_ep = 0
for epoch in range(cfg["epochs"]):
    n.train()
    start = time.time()
    for i, (X, y_t) in enumerate(dl_train):
        opt.zero_grad()
        y_out = n(X.float().to(device)).to("cpu")
        y_t = y_t.float()
        tot_loss = None
        for l in cfg["losses"]:
            y_l = l["pre_proc_out"](y_out)
            y_t_l = l["pre_proc_y"](y_t)
            loss = l["l_func"](y_l, y_t_l)*l["weight"]
            writer.add_scalar("Loss/{}".format(l["name"]), loss, epoch*j + i)
            if not tot_loss: tot_loss=loss
            else: tot_loss += loss
        tot_loss.backward()
        opt.step()
    j = i
    print("Steps:", j)
    print("Time for an epoch:", time.time() - start)
    scheduler.step()
    lr = scheduler.get_lr()
    writer.add_scalar("Stats/Lr", lr[0], epoch*j + i)

    with torch.no_grad():
      n.eval()
      val_loss = 0
      cos_losses = 0
      for k, (X, y_t) in enumerate(dl_val):
          y_out = n(X.cuda())
          cos_losses += cos_loss(y_out.float().cpu()[:,:3], y_t.float().cpu()[:,:3])
      writer.add_scalar("Val_Loss/cos", cos_losses/k, epoch)
