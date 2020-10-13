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
    return torch.mean(-cos_sim(x, y))


model_hp_cnn = {"filters" : [32, 64, 128, 128, 16],
"kernels" : [5, 3, 5, 3, 3],
 "dim_s" : [2,1,2,1,1],
 "activation" : "selu",
 "drop_p" : 0.1}

model_hp_fcn = {"layer_dims" : [256, 256, 4],
 "activation" : "selu",
 "drop_p" : 0.1}

y_mult = torch.Tensor([1e0]).view((1,1)).to("cpu")
losses = [{"name" : "MSE", "l_func": torch.nn.MSELoss(), "weight" : 0.005, "pre_proc_y": lambda x: x*y_mult, "pre_proc_out" : lambda x:x},
          {"name" : "L1", "l_func": torch.nn.L1Loss(), "weight" : 0.05, "pre_proc_y": lambda x: x*y_mult, "pre_proc_out" : lambda x:x},
          {"name" : "cos_sim", "l_func": cos_loss, "weight" : 10., "pre_proc_y" : lambda x: x[:,:3]*y_mult, "pre_proc_out" : lambda x: x[:,:3]}]

cfg = {"model" : "fcn",
        "model_hp" : model_hp_fcn,
      "optimizer" : optim.Yogi,
      "data_folder" : cwd + "/data/keyp_RGB.npy",
      "y_path": cwd + "/data/IR_150k_cropped/vel_optimal_action_in_eef_array.npy",
      "data" : "KP_data",
      "validation_split" : 0.2,
      "batch_size" : 64,
      "lr" : 3e-4,
      "losses" : losses,
      "losses_names_weights" : sorted([[losses[i]["name"], losses[i]["weight"]] for i in range(len(losses))]),
      "device" : device,
      "epochs" : 50,
      "sch_step" : 5,
      "sch_gamma" : 0.9
      }
