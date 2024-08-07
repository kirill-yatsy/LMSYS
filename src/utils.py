import logging
from argparse import ArgumentParser
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import ignite.distributed as idist
import torch
from ignite.contrib.engines import common
from ignite.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.handlers.early_stopping import EarlyStopping
from ignite.utils import setup_logger
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer

# from data import setup_data
import pandas as pd

def get_default_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "config", type=Path, help="Config file path", default="config.yaml"
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=["nccl", "gloo"],
        type=str,
        help="DDP backend",
    )
    return parser


def setup_config(parser=None):
    if parser is None:
        parser = get_default_parser()

    args = parser.parse_args()
    config_path = args.config
    config = OmegaConf.load(config_path)
    config.backend = args.backend

    return config


def log_metrics(engine: Engine, tag: str) -> None:
    """Log `engine.state.metrics` with given `engine` and `tag`.

    Parameters
    ----------
    engine
        instance of `Engine` which metrics to log.
    tag
        a string to add at the start of output.
    """
    # metrics_format = "{0} [{1}/{2}]: {3}".format(
    #     tag, engine.state.epoch, engine.state.iteration, engine.state.metrics
    # )
    # engine.logger.info(metrics_format)


def resume_from(
    to_load: Mapping,
    checkpoint_fp: Union[str, Path],
    logger: Logger,
    strict: bool = True,
    model_dir: Optional[str] = None,
) -> None:
    """Loads state dict from a checkpoint file to resume the training.

    Parameters
    ----------
    to_load
        a dictionary with objects, e.g. {“model”: model, “optimizer”: optimizer, ...}
    checkpoint_fp
        path to the checkpoint file
    logger
        to log info about resuming from a checkpoint
    strict
        whether to strictly enforce that the keys in `state_dict` match the keys
        returned by this module’s `state_dict()` function. Default: True
    model_dir
        directory in which to save the object
    """
    if isinstance(checkpoint_fp, str) and checkpoint_fp.startswith("https://"):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_fp,
            model_dir=model_dir,
            map_location="cpu",
            check_hash=True,
        )
    else:
        if isinstance(checkpoint_fp, str):
            checkpoint_fp = Path(checkpoint_fp)

        if not checkpoint_fp.exists():
            raise FileNotFoundError(f"Given {str(checkpoint_fp)} does not exist.")
        checkpoint = torch.load(checkpoint_fp, map_location="cpu")

    Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint, strict=strict)
    logger.info("Successfully resumed from a checkpoint: %s", checkpoint_fp)


def setup_output_dir(config: Any, rank: int) -> Path:
    """Create output folder."""
    output_dir = config.output_dir
    if rank == 0:
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{now}-backend-{config.backend}-lr-{config.lr}"
        path = Path(config.output_dir, name)
        path.mkdir(parents=True, exist_ok=True)
        output_dir = path.as_posix()
    return Path(idist.broadcast(output_dir, src=0))


def save_config(config, output_dir):
    """Save configuration to config-lock.yaml for result reproducibility."""
    with open(f"{output_dir}/config-lock.yaml", "w") as f:
        OmegaConf.save(config, f)


def setup_logging(config: Any) -> Logger:
    """Setup logger with `ignite.utils.setup_logger()`.

    Parameters
    ----------
    config
        config object. config has to contain `verbose` and `output_dir` attribute.

    Returns
    -------
    logger
        an instance of `Logger`
    """
    green = "\033[32m"
    reset = "\033[0m"
    logger = setup_logger(
        name=f"{green}[ignite]{reset}",
        level=logging.DEBUG if config.debug else logging.INFO,
        filepath=config.output_dir / "training-info.log",
    )
    return logger


def setup_exp_logging(config, trainer, optimizers, evaluators):
    """Setup Experiment Tracking logger from Ignite."""
    logger = common.setup_tb_logging(
        config.output_dir,
        trainer,
        optimizers,
        evaluators,
        config.log_every_iters,
    )
    return logger


def setup_handlers(
    trainer: Engine,
    evaluator: Engine,
    config: Any,
    to_save_train: Optional[dict] = None,
    to_save_eval: Optional[dict] = None,
):
    """Setup Ignite handlers."""

    ckpt_handler_train = ckpt_handler_eval = None
    # checkpointing
    saver = DiskSaver(config.output_dir / "checkpoints", require_empty=False)
    ckpt_handler_train = Checkpoint(
        to_save_train,
        saver,
        filename_prefix=config.filename_prefix,
        n_saved=config.n_saved,
    )

   
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=config.save_every_iters),
        ckpt_handler_train,
    )
    global_step_transform = None
    if to_save_train.get("trainer", None) is not None:
        global_step_transform = global_step_from_engine(to_save_train["trainer"])
    ckpt_handler_eval = Checkpoint(
        to_save_eval,
        saver,
        filename_prefix="best",
        n_saved=config.n_saved,
        global_step_transform=global_step_transform,
        score_name="eval_accuracy",
        score_function=Checkpoint.get_default_score_fn("Accuracy"),
    ) 

    evaluator.add_event_handler(Events.EPOCH_COMPLETED(every=1), ckpt_handler_eval)

    # early stopping
    def score_fn(engine: Engine):
        return engine.state.metrics["Accuracy"]

    es = EarlyStopping(config.patience, score_fn, trainer)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, es)
    return ckpt_handler_train, ckpt_handler_eval


def thresholded_output_transform(output):
    y_pred, y = output
    return torch.round(torch.sigmoid(y_pred)), y


def build_prompt(instruction, input, resp1, resp2):
    rsp = f"### Response 1:\n{resp1}\n\n### Response 2:\n{resp2}"

    input_sequence = f"Below are two responses for a given task. The task is defined by the Instruction with an Input that provides further context. Evaluate the responses and generate a reference answer for the task.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n{rsp}\n\n### Evaluation:\n"

    return input_sequence
    

def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
 
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "</s>"
    DEFAULT_UNK_TOKEN = "</s>"
    assert tokenizer.pad_token == '[PAD]'
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )

    tokenizer.pad_token_id = 0  # unk

    return tokenizer


def build_prompt2(tokenizer, max_tokens, instruction, input, resp1, resp2):
    def truncate_text(text, max_tokens, tokenizer):
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_tokens)
  
        return tokenizer.decode(tokens, skip_special_tokens=True)

    # validate input, resp1, resp2 on nan
    if pd.isna(input):
        input = ""
    if pd.isna(resp1):
        resp1 = ""
    if pd.isna(resp2):
        resp2 = ""

    rsp = f"### Response 1:\n{resp1}\n\n### Response 2:\n{resp2}"

    input_sequence = build_prompt(instruction, "", "", "")
     
    # Calculate remaining tokens after reserving space for non-truncatable parts
    input_tokens = tokenizer.encode(input_sequence)
 
    reserved_tokens = len(input_tokens)
    remaining_tokens = max_tokens - reserved_tokens

    # Calculate the number of tokens that can be allocated to input, resp1, and resp2
    input_tokens_count = len(tokenizer.encode(input))
    resp1_tokens_count = len(tokenizer.encode(resp1))
    resp2_tokens_count = len(tokenizer.encode(resp2))

    final_input = input
    final_resp1 = resp1
    final_resp2 = resp2

    tokenized_prompt = tokenizer.encode(build_prompt(instruction, final_input, final_resp1, final_resp2))
    total_tokens_count = len(tokenized_prompt)
    
    alpha = 0.0

    while (len(tokenized_prompt) > max_tokens):
        input_ratio = input_tokens_count / total_tokens_count - alpha
        resp1_ratio = resp1_tokens_count / total_tokens_count - alpha
        resp2_ratio = resp2_tokens_count / total_tokens_count - alpha

        max_input_tokens = int(remaining_tokens * input_ratio)
        max_resp1_tokens = int(remaining_tokens * resp1_ratio)
        max_resp2_tokens = int(remaining_tokens * resp2_ratio)

        final_input = truncate_text(input, max_input_tokens, tokenizer)
        final_resp1 = truncate_text(resp1, max_resp1_tokens, tokenizer)
        final_resp2 = truncate_text(resp2, max_resp2_tokens, tokenizer)
        tokenized_prompt = tokenizer.encode(build_prompt(instruction, final_input, final_resp1, final_resp2))
        alpha += 0.1

        if alpha > 1.0:
            print("alpha > 1.0. Breaking loop.")
            break

    return build_prompt(instruction, final_input, final_resp1, final_resp2)
        
    

# if __name__ == "__main__":
#     from omegaconf import OmegaConf

#     config = OmegaConf.load("config.yaml")
#     dataloader_train, dataloader_eval = setup_data(config)
#     tokenizer = get_tokenizer(config.model)
#     decodedl = []
#     # show progress bar
#     for i, batch in tqdm(enumerate(dataloader_train)):
#         decoded = tokenizer.decode(batch["input_ids"].squeeze())
#         decodedl.append(decoded)
    
#     # create df from decoded
#     df = pd.DataFrame(decodedl)

#     df.to_csv("./decoded.csv", index=False)
         
 
