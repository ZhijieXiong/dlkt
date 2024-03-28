"""
This code is copied from: https://github.com/microsoft/ToRA
"""
import argparse
from transformers import SchedulerType

from fintune import finetune


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="The name of the dataset to use (via the datasets library).")
    parser.add_argument("--dataset_config_name", type=str, default=None,
                        help="The configuration name of the dataset to use (via the datasets library).")
    parser.add_argument("--train_file", type=str, default=None,
                        help="A csv or a json file containing the training data.")
    parser.add_argument("--model_name_or_path", type=str, required=False,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--config_name", type=str, default=None,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--use_lora", action="store_true",
                        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.")
    parser.add_argument("--lora_rank", type=int, default=64,
                        help="The rank of lora.")
    parser.add_argument("--lora_alpha", type=float, default=16,
                        help="The alpha parameter of lora.")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="The dropout rate of lora modules.")
    parser.add_argument("--save_merged_lora_model", action="store_true",
                        help="If passed, will merge the lora modules and save the entire model.")
    parser.add_argument("--use_flash_attn", action="store_true",
                        help="If passed, will use flash attention to train the model.")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--use_slow_tokenizer", action="store_true",
                        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="The maximum total sequence length (prompt+completion) of each training example.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
                        help="The scheduler type to use.",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--warmup_ratio", type=float, default=0,
                        help="Ratio of total training steps used for warmup.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None,
                        help="A seed for reproducible training.")
    parser.add_argument("--preprocessing_num_workers", type=int, default=None,
                        help="The number of processes to use for the preprocessing.")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--checkpointing_steps", type=str, default=None,
                        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument("--logging_steps", type=int, default=None,
                        help="Log the training loss and learning rate every logging_steps steps.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="If the training should continue from a checkpoint folder.")
    parser.add_argument("--with_tracking", action="store_true",
                        help="Whether to enable experiment trackers for logging.")
    parser.add_argument("--report_to", type=str, default="all",
                        help=(
                            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
                            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
                            "Only applicable when `--with_tracking` is passed."
                        ))
    parser.add_argument("--low_cpu_mem_usage", action="store_true",
                        help=(
                            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                            "If passed, LLM loading time and RAM consumption will be benefited."
                        ))
    parser.add_argument("--mask_prompt", action="store_true")

    args = parser.parse_args()
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["json", "jsonl"], "`train_file` should be a json/jsonl file."
    params = vars(args)

    finetune(params)
