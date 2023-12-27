import torch
import torch.nn as nn
import torch.optim as optim

from .KnowledgeTracingTrainer import KnowledgeTracingTrainer
from .LossRecord import LossRecord
from ..model.Model4LongTail import *


class MutualEnhance4LongTailTrainer(KnowledgeTracingTrainer):
    def __init__(self, params, objects):
        super(MutualEnhance4LongTailTrainer, self).__init__(params, objects)

