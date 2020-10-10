import torch
import torch.nn as nn
import torch_optimizer as optim

import numpy as np
import matplotlib.pyplot as plt
import time
import os

cwd = os.getcwd()



device = "cuda"

cos_sim = nn.CosineSimilarity(dim = 1)
def cos_loss(x, y):
    print("x",x)
    print("y",y)
    return torch.mean(-cos_sim(x, y))


model_hp = {"filters" : [32, 64, 128, 128, 16],
"kernels" : [5, 3, 5, 3, 3],
 "dim_s" : [2,1,2,1,1],
 "activation" : "selu",
 "drop_p" : 0.1}

y_mult = torch.Tensor([1e4, 1e4, 1e3, 1e3]).view((1,4)).to("cpu")
losses = [{"name" : "MSE", "l_func": torch.nn.MSELoss(), "weight" : 0.0005, "pre_proc_y": lambda x: x*y_mult, "pre_proc_out" : lambda x:x},
          {"name" : "L1", "l_func": torch.nn.L1Loss(), "weight" : 0.005, "pre_proc_y": lambda x: x*y_mult, "pre_proc_out" : lambda x:x},
          {"name" : "cos_sim", "l_func": cos_loss, "weight" : 10., "pre_proc_y" : lambda x: x[:,:3], "pre_proc_out" : lambda x: x[:,:3]}]

cfg = {"model" : "cnn",
        "model_hp" : model_hp,
      "optimizer" : optim.Yogi,
      "data_folder" : cwd + "/data/IR_150k_cropped/IR1/",
      "y_path": cwd + "/data/IR_150k_cropped/vel_optimal_action_in_eef_array.npy",
      "data" : "IR_data",
      "validation_split" : 0.2,
      "batch_size" : 8,
      "losses" : losses,
      "device" : device,
      "epochs" : 50,
      "sch_step" : 5,
      "sch_gamma" : 3
      }
