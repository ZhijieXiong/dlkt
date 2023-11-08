import torch
import random
import numpy as np


def set_seed(seed):
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        print("Set seed failed, details are ", e)
        pass
    np.random.seed(seed)
    random.seed(seed)
