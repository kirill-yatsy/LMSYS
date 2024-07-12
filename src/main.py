import os
from pprint import pformat
from typing import Any
import sys
import ignite.distributed as idist 
from data import setup_data
from ignite.engine import Events
from ignite.handlers import CosineAnnealingScheduler
from ignite.metrics import Accuracy, Loss, Fbeta, Precision, Recall
from ignite.utils import manual_seed
from models import TransformerModel
from torch import nn, optim
from trainers import setup_evaluator, setup_trainer
from utils import *
from accelerate import Accelerator

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # remove tokenizer paralleism warning
import bitsandbytes as bnb

from ignite.contrib.handlers.tqdm_logger import ProgressBar

class ProgressBarCustom(ProgressBar):
    """To force tqdm to overwrite previous line"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from tqdm import tqdm as tqdm_base

        def tqdm_custom(*args, **kwargs):
            if hasattr(tqdm_base, '_instances'):
                for instance in list(tqdm_base._instances):
                    tqdm_base._decr_instances(instance)
            return tqdm_base(*args, **kwargs)

        self.pbar_cls = tqdm_custom




def run(local_rank: int, config: Any):


    accelerator = Accelerator( )

    # make a certain seed
    rank = idist.get_rank()
    manual_seed(config.seed + rank)

    # create output folder and copy config file to output dir
    output_dir = setup_output_dir(config, rank)
    if rank == 0:
        save_config(config, output_dir)

    config.output_dir = output_dir

    # donwload datasets and create dataloaders
    dataloader_train, dataloader_eval = setup_data(config)

    config.num_iters_per_epoch = len(dataloader_train)

    # model, optimizer, loss function, device
    device = idist.device()
    # model = idist.auto_model(
        
    # )
    model = TransformerModel(config) 
    

    config.lr *= idist.get_world_size()
    optimizer = idist.auto_optim(
        bnb.optim.Adam8bit(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    )
    loss_fn = nn.BCEWithLogitsLoss().to(device=device)

    le = config.num_iters_per_epoch
    
    lr_scheduler = CosineAnnealingScheduler(
        optimizer, "lr", config.lr, config.lr * 10**-4, len(dataloader_train) * config.max_epochs
    )

    model, optimizer, dataloader_train, lr_scheduler = accelerator.prepare(model, optimizer, dataloader_train, lr_scheduler)


    # setup metrics to attach to evaluator
    metrics = {
        "Accuracy": Accuracy(output_transform=thresholded_output_transform, is_multilabel=True),
        "Precision": Precision(output_transform=thresholded_output_transform, is_multilabel=True ),
        "Recall": Recall(output_transform=thresholded_output_transform, is_multilabel=True),
        "Fbeta": Fbeta(beta=1.0, output_transform=thresholded_output_transform ),
        "Average_Precision": Precision(average=True, output_transform=thresholded_output_transform, is_multilabel=True ),
        "Average_Recall": Recall(average=True, output_transform=thresholded_output_transform, is_multilabel=True),
        "Average_Fbeta": Fbeta(average=True, beta=1.0, output_transform=thresholded_output_transform ),
        "Loss": Loss(loss_fn),
    }

    # trainer and evaluator
    trainer = setup_trainer(
        config, model, optimizer, loss_fn, device, dataloader_train.sampler, accelerator
    )
    evaluator = setup_evaluator(config, model, metrics, device)

    # setup engines logger with python logging
    # print training configurations
    logger = setup_logging(config)
    logger.info("Configuration: \n%s", pformat(config))
    trainer.logger = evaluator.logger = logger

    trainer.add_event_handler(Events.ITERATION_COMPLETED, lr_scheduler)

    # setup ignite handlers
    to_save_train = {
        "model": model,
        "optimizer": optimizer,
        "trainer": trainer,
        "lr_scheduler": lr_scheduler,
    }
    to_save_eval = {"model": model}
    ckpt_handler_train, ckpt_handler_eval = setup_handlers(
        trainer, evaluator, config, to_save_train, to_save_eval
    )

    checkpoint_fp = config.checkpoint_fp
    if checkpoint_fp:
        ckpt_handler_train.load_objects(to_load=to_save_train, checkpoint=config.checkpoint_fp) 

    # experiment tracking
    if rank == 0:
        exp_logger = setup_exp_logging(config, trainer, optimizer, evaluator)

    # print metrics to the stderr
    # with `add_event_handler` API
    # for training stats
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=config.log_every_iters),
        log_metrics,
        tag="train",
    )

    ProgressBarCustom(persist=False, desc="Training").attach(trainer)
    ProgressBarCustom(persist=False, desc="Validation").attach(evaluator)

    # run evaluation at every training epoch end
    # with shortcut `on` decorator API and
    # print metrics to the stderr
    # again with `add_event_handler` API
    # for evaluation stats
    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def _():
        evaluator.run(dataloader_eval, epoch_length=config.eval_epoch_length)
        log_metrics(evaluator, "eval")

 
    # let's try run evaluation first as a sanity check
    # @trainer.on(Events.STARTED)
    # def _():
    #     evaluator.run(dataloader_eval, epoch_length=config.eval_epoch_length)

    # setup if done. let's run the training
    trainer.run(
        dataloader_train,
        max_epochs=config.max_epochs,
        epoch_length=config.train_epoch_length,
    )

    # close logger
    if rank == 0:
        exp_logger.close()

    # show last checkpoint names
    logger.info(
        "Last training checkpoint name - %s",
        ckpt_handler_train.last_checkpoint,
    )

    logger.info(
        "Last evaluation checkpoint name - %s",
        ckpt_handler_eval.last_checkpoint,
    )


# main entrypoint
def main():

    config = setup_config()
    with idist.Parallel(config.backend) as p:
        p.run(run, config=config)


if __name__ == "__main__":
    sys.argv = ["src/main.py", "config.yaml"]
    main()
