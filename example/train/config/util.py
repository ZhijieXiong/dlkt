import json


def config_optimizer(local_params, global_params, global_objects, model_name="kt_model", same_as_kt=False):
    # 优化器配置
    if same_as_kt:
        optimizer_type = local_params[f"optimizer_type"]
        weight_decay = local_params[f"weight_decay"]
        momentum = local_params[f"momentum"]
        learning_rate = local_params[f"learning_rate"]
        enable_lr_schedule = local_params[f"enable_lr_schedule"]
        lr_schedule_type = local_params[f"lr_schedule_type"]
        lr_schedule_step = local_params[f"lr_schedule_step"]
        lr_schedule_milestones = eval(local_params[f"lr_schedule_milestones"])
        lr_schedule_gamma = local_params[f"lr_schedule_gamma"]
        enable_clip_grad = local_params[f"enable_clip_grad"]
        grad_clipped = local_params[f"grad_clipped"]
    else:
        optimizer_type = local_params[f"optimizer_type{'' if (model_name == 'kt_model') else ('_' + model_name)}"]
        weight_decay = local_params[f"weight_decay{'' if (model_name == 'kt_model') else ('_' + model_name)}"]
        momentum = local_params[f"momentum{'' if (model_name == 'kt_model') else ('_' + model_name)}"]
        learning_rate = local_params[f"learning_rate{'' if (model_name == 'kt_model') else ('_' + model_name)}"]
        enable_lr_schedule = local_params[f"enable_lr_schedule{'' if (model_name == 'kt_model') else ('_' + model_name)}"]
        lr_schedule_type = local_params[f"lr_schedule_type{'' if (model_name == 'kt_model') else ('_' + model_name)}"]
        lr_schedule_step = local_params[f"lr_schedule_step{'' if (model_name == 'kt_model') else ('_' + model_name)}"]
        lr_schedule_milestones = eval(local_params[f"lr_schedule_milestones{'' if (model_name == 'kt_model') else ('_' + model_name)}"])
        lr_schedule_gamma = local_params[f"lr_schedule_gamma{'' if (model_name == 'kt_model') else ('_' + model_name)}"]
        enable_clip_grad = local_params[f"enable_clip_grad{'' if (model_name == 'kt_model') else ('_' + model_name)}"]
        grad_clipped = local_params[f"grad_clipped{'' if (model_name == 'kt_model') else ('_' + model_name)}"]

    global_params["optimizers_config"][model_name] = {}
    optimizer_config = global_params["optimizers_config"][model_name]
    optimizer_config["type"] = optimizer_type
    optimizer_config[optimizer_type] = {}
    optimizer_config[optimizer_type]["lr"] = learning_rate
    optimizer_config[optimizer_type]["weight_decay"] = weight_decay
    if optimizer_type == "sgd":
        optimizer_config[optimizer_type]["momentum"] = momentum

    global_params["schedulers_config"][model_name] = {}
    scheduler_config = global_params["schedulers_config"][model_name]
    if enable_lr_schedule:
        scheduler_config["use_scheduler"] = True
        scheduler_config["type"] = lr_schedule_type
        scheduler_config[lr_schedule_type] = {}
        if lr_schedule_type == "StepLR":
            scheduler_config[lr_schedule_type]["step_size"] = lr_schedule_step
            scheduler_config[lr_schedule_type]["gamma"] = lr_schedule_gamma
        elif lr_schedule_type == "MultiStepLR":
            scheduler_config[lr_schedule_type]["milestones"] = lr_schedule_milestones
            scheduler_config[lr_schedule_type]["gamma"] = lr_schedule_gamma
        else:
            raise NotImplementedError()
    else:
        scheduler_config["use_scheduler"] = False

    global_params["grad_clip_config"][model_name] = {}
    grad_clip_config = global_params["grad_clip_config"][model_name]
    grad_clip_config["use_clip"] = enable_clip_grad
    if enable_clip_grad:
        grad_clip_config["grad_clipped"] = grad_clipped

    global_objects["logger"].info(
        f"    model optimized: {model_name}, optimizer type: {optimizer_type}, {optimizer_type} config: {json.dumps(optimizer_config[optimizer_type])}, "
        f"use lr schedule: {enable_lr_schedule}{f', schedule type is {lr_schedule_type}: {json.dumps(scheduler_config[lr_schedule_type])}' if enable_lr_schedule else ''}, "
        f"use clip for grad: {enable_clip_grad}{f', norm clipped: {grad_clipped}' if enable_clip_grad else ''}"
    )
