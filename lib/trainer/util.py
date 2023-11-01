import torch.optim as optim


def create_optimizer(opt_config, parameters):
    optimizer_type = opt_config["optimizer_type"]
    learning_rate = opt_config["learning_rate"]
    weight_decay = opt_config.get("weight_decay", 0.0)
    momentum = opt_config.get("momentum", 0.0)
    eps = 1e-8

    if optimizer_type == 'adam':
        optimizer = optim.Adam(parameters, learning_rate, weight_decay=weight_decay, eps=eps)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(parameters, learning_rate, weight_decay=weight_decay, momentum=momentum)
    else:
        optimizer = optim.Adam(parameters, learning_rate, weight_decay=weight_decay)

    return optimizer
