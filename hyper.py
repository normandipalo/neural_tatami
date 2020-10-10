import torch
import torch.nn as nn
import torch_optimizer as optim

import numpy as np
import matplotlib.pyplot as plt
import time

device = "cuda"

cos_sim = nn.CosineSimilarity(dim = 1)
def cos_loss(x, y):
    return torch.mean(-cos_sim(x, y))


model_hp = {"filters" : [32, 32, 64, 128, 1],
"kernels" : [5, 3, 3, 3, 1],
 "dim_s" : [1,1,2,2,1],
 "activation" : "selu",
 "drop_p" : 0.1}

losses = [{"name" : "MSE", "l_func": torch.nn.MSELoss(), "weight" : 0.0, "pre_proc": lambda x: x*torch.Tensor([1e4, 1e4, 1e3, 1e3]).to(device)},
          {"name" : "L1", "l_func": torch.nn.L1Loss(), "weight" : 0.01, "pre_proc": lambda x: x*torch.Tensor([1e4, 1e4, 1e3, 1e3]).to(device)},
          {"name" : "cos_sim", "l_func": cos_loss, "weight" : 10., "pre_proc" : lambda x: x[:,:3]}]

cfg = {"model" : "cnn",
        "model_hp" : model_hp,
      "optimizer" : optim.Yogi,
      "data_folder" : "/home/norman/Downloads/IR_150k_cropped/IR1/",
      "data" : "IR_data",
      "validation_split" : 0.2,
      "batch_size" : 32,
      "losses" : losses,
      "device" : device,
      "epochs" : 50,
      }
