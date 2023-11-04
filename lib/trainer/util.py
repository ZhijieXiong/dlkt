import torch.optim as optim


def create_optimizer(parameters, opt_config):
    if opt_config["type"] == 'sgd':
        optimizer = optim.SGD(parameters, **opt_config["sgd"])
    else:
        optimizer = optim.Adam(parameters, **opt_config["adam"])
    return optimizer


def create_scheduler(optimizer, sch_config):
    if sch_config["type"] == "CosineAnnealingLR":
        scheduler = None
    else:
        # 默认StepLR
        scheduler = optim.lr_scheduler.StepLR(optimizer, **sch_config["StepLR"])
    return scheduler
