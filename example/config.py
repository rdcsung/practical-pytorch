"""This file is to provide common config value"""

import torch

HOST_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")