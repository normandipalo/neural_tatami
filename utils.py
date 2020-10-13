import numpy as np
import matplotlib.pyplot as plt
import time
import hashlib
import os
import json
import copy

from hyper import cfg, cos_loss
from models import *
from data import *


def store_hp(cfg):
    # I create a deterministic hash based on the content of the cfg and model_hp dictionaries.
    # Dictionaries generally get different hashes at every run. I need to be sure that there
    # are no nested dictionaries to get a deterministic output. Hence, I remove model_hp for now.
    new_cfg = copy.deepcopy(cfg)
    model_hp = cfg["model_hp"]
    new_cfg.pop("model_hp")
    new_cfg.pop("losses")
    #hp_hash = hashlib.sha1(str(json.dumps(cfg, sort_keys=True)).encode('utf-8'))
    hp_hash_cfg = hashlib.sha1(repr(sorted(new_cfg.items())).encode('utf-8'))
    hp_hash_mod_hp = hashlib.sha1(repr(sorted(model_hp.items())).encode('utf-8'))
    hex_dig_cfg = hp_hash_cfg.hexdigest()
    hex_dig_mod_hp = hp_hash_mod_hp.hexdigest()
    print(hex_dig_cfg[:4] + hex_dig_mod_hp[:4])
    hash = hex_dig_cfg[:4] + hex_dig_mod_hp[:4]
    with open(os.getcwd() + "/hp_logs/" + str(hash) + ".txt", "w") as file:
        file.write(str(model_hp))
        file.write(str(cfg))
    return hash
