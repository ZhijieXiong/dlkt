import logging
import math
import os
import random
import datasets
import torch
import transformers
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
)


from llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from util import *


def finetune(params):
    # A hacky way to make llama work with flash attention
    if params["use_flash_attn"]:
        replace_llama_attn_with_flash_attn()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}
    if params["with_tracking"]:
        accelerator_log_kwargs["log_with"] = params["report_to"]
        accelerator_log_kwargs["project_dir"] = params["output_dir"]
    accelerator = Accelerator(gradient_accumulation_steps=params["gradient_accumulation_steps"],
                              **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.logging.set_verbosity_error()

    if params["seed"] is not None:
        # If passed along, set the training seed now.
        set_seed(params["seed"])

    if accelerator.is_main_process:
        if params["output_dir"] is not None:
            os.makedirs(params["output_dir"], exist_ok=True)

    accelerator.wait_for_everyone()

    data_files = {}
    dataset_args = {}
    if params["train_file"] is not None:
        data_files["train"] = params["train_file"]
    raw_datasets = load_dataset("json", data_files=data_files, **dataset_args)

    # Load pretrained model and tokenizer
    if params["config_name"]:
        config = AutoConfig.from_pretrained(params["config_name"])
    elif params["model_name_or_path"]:
        config = AutoConfig.from_pretrained(params["model_name_or_path"])
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    if params["tokenizer_name"]:
        tokenizer = AutoTokenizer.from_pretrained(params.tokenizer_name, use_fast=not params.use_slow_tokenizer)
    elif params["model_name_or_path"]:
        tokenizer = AutoTokenizer.from_pretrained(params.model_name_or_path, use_fast=not params.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if params["model_name_or_path"]:
        model = AutoModelForCausalLM.from_pretrained(
            params["model_name_or_path"],
            from_tf=bool(".ckpt" in params["model_name_or_path"]),
            config=config,
            low_cpu_mem_usage=params["low_cpu_mem_usage"],
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    if params["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    tokenizer.pad_token = tokenizer.unk_token

    if params["use_lora"]:
        logger.info("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=params["lora_rank"],
            lora_alpha=params["lora_alpha"],
            lora_dropout=params["lora_dropout"]
        )
        model = get_peft_model(model, peft_config)
        print("Trainable parameters:")
        model.print_trainable_parameters()

    print("Is prompt masked:", params["mask_prompt"])
    if "prompt" in raw_datasets["train"].column_names and "completion" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=params["max_seq_length"],
            mask_prompt=params["mask_prompt"],
        )
    elif "messages" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=params["max_seq_length"],
            mask_prompt=params["mask_prompt"],
        )
    else:
        raise ValueError("You need to have either 'prompt'&'completion' or 'messages' in your column names.")

    with accelerator.main_process_first():
        lm_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=params["preprocessing_num_workers"],
            load_from_cache_file=not params["overwrite_cache"],
            remove_columns=[name for name in raw_datasets["train"].column_names if
                            name not in ["input_ids", "labels", "attention_mask"]],
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets.set_format(type="pt")
        lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())

    train_dataset = lm_datasets["train"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=params["per_device_train_batch_size"]
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": params["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=params["learning_rate"])

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / params.gradient_accumulation_steps)
    if params.max_train_steps is None:
        params.max_train_steps = params.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = params["max_train_steps"] if overrode_max_train_steps else params["max_train_steps"] * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=params["lr_scheduler_type"],
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * params["warmup_ratio"]),
    )

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / params["gradient_accumulation_steps"])
    if overrode_max_train_steps:
        params["max_train_steps"] = params["num_train_epochs"] * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    params.num_train_epochs = math.ceil(params["max_train_steps"] / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = params.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if params["with_tracking"]:
        experiment_config = deepcopy(params)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("tora", experiment_config)

    # Train!
    total_batch_size = params["per_device_train_batch_size"] * accelerator.num_processes * params["gradient_accumulation_steps"]

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {params['num_train_epochs']}")
    logger.info(f"  Instantaneous batch size per device = {params['per_device_train_batch_size']}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {params['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {params['max_train_steps']}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(params["max_train_steps"]), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if params["resume_from_checkpoint"]:
        if params['resume_from_checkpoint'] is not None or params.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {params['resume_from_checkpoint']}")
            accelerator.load_state(params["resume_from_checkpoint"])
            path = os.path.basename(params["resume_from_checkpoint"])
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * params["gradient_accumulation_steps"]
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    for epoch in range(starting_epoch, params["num_train_epochs"]):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if params.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and completed_steps < resume_step:
                    if step % params.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                # # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if params["logging_steps"] and completed_steps % params["logging_steps"] == 0:
                    avg_loss = accelerator.gather(
                        total_loss).mean().item() / params["gradient_accumulation_steps"] / params["logging_steps"]
                    logger.info(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                    if params["with_tracking"]:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                            },
                            step=completed_steps,
                        )
                    total_loss = 0

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if params.output_dir is not None:
                            output_dir = os.path.join(params.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                if completed_steps >= params.max_train_steps:
                    break

        if params["checkpointing_steps"] == "epoch":
            output_dir = f"epoch_{epoch}"
            if params["output_dir"] is not None:
                output_dir = os.path.join(params["output_dir"], output_dir)
            accelerator.save_state(output_dir)

    if params["with_tracking"]:
        accelerator.end_training()

    if params.output_dir is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            tokenizer.save_pretrained(params.output_dir)
        unwrapped_model = accelerator.unwrap_model(model)
        # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
        # Otherwise, sometimes the model will be saved with only part of the parameters.
        # Also, accelerator needs to use the wrapped model to get the state_dict.
        state_dict = accelerator.get_state_dict(model)
        if params["use_lora"]:
            # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process
            # and has its own save_pretrained function for only saving lora modules.
            # We have to mannually specify the is_main_process outside the save_pretrained function.
            if accelerator.is_main_process:
                unwrapped_model.save_pretrained(params.output_dir, state_dict=state_dict)
        else:
            unwrapped_model.save_pretrained(
                params["output_dir"],
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=state_dict
            )