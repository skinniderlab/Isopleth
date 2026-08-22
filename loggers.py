import math
import os
import pandas as pd
import torch
from tqdm import tqdm


class NaNLossError(Exception):
    """Raised when NaN loss is produced during training"""
    pass

class EarlyStopping():
    """
    Monitor the training process to stop training early if the model shows
    evidence of beginning to overfit the validation dataset, and save the
    best model.

    Uses exponential moving average (EMA) of validation loss to smooth out
    noise and make early stopping more robust.

    Note that patience here is measured in steps, rather than in epochs,
    because the size of an epoch will not be consistent if the size of the
    dataset changes.

    Inspired by/adapted from:
    https://github.com/Bjarten/early-stopping-pytorch
    https://github.com/fastai/fastai/blob/master/courses/dl2/imdb_scripts/finetune_lm.py
    """

    def __init__(self, patience=100, ema_alpha=0.1):
        """
        Args:
            patience: (int) if the validation loss fails to improve for this
              number of consecutive batches, training will be stopped
            ema_alpha: (float) smoothing factor for exponential moving average.
              Higher values = more weight on recent observations (less smoothing).
              Default 0.1 gives effective window of ~10 steps. Set to 1.0 to
              disable smoothing and use raw validation loss.
        """
        self.patience = patience
        self.ema_alpha = ema_alpha
        self.counter = 0
        self.best_loss = None
        self.curr_loss = None
        self.step_at_best = 0
        self.stop = False
        print(f"instantiated early stopping with patience={self.patience}, "
              f"ema_alpha={self.ema_alpha}")

    def __call__(self, val_loss, model, output_file, step_idx):
        # do nothing if early stopping is disabled
        if self.patience > 0:
            # if NaN loss, raise error
            if math.isnan(val_loss):
                raise NaNLossError(f"NaN loss detected at step {step_idx}")

            # update loss based on exponential moving average
            if self.curr_loss is None:
                self.curr_loss = val_loss
            else:
                self.curr_loss = (self.ema_alpha * val_loss +
                                 (1 - self.ema_alpha) * self.curr_loss)

            if self.best_loss is None:
                self.best_loss = self.curr_loss
                self.step_at_best = step_idx
                self.save_model(model, output_file)
            elif self.curr_loss >= self.best_loss:
                # loss is not decreasing
                self.counter += 1
                if self.counter >= self.patience:
                    self.stop = True
                    print(f"stopping early with best EMA loss {self.best_loss:.4f} "
                          f"at step {self.step_at_best}")
            else:
                # loss is decreasing
                self.best_loss = self.curr_loss
                self.step_at_best = step_idx
                # reset counter
                self.counter = 0
                self.save_model(model, output_file)

    def save_model(self, model, output_file):
        torch.save(model.state_dict(), output_file)

def log_loss(output_file, epoch, batch_idx, training_loss, validation_loss):
    sched = pd.DataFrame({'epoch': epoch + 1,
                          'minibatch': batch_idx,
                          'outcome': ['training loss', 'validation loss'],
                          'value': [training_loss, validation_loss]})

    # write training schedule (write header if file does not exist)
    if not os.path.isfile(output_file) or batch_idx == 0:
        sched.to_csv(output_file, index=False)
    else:
        sched.to_csv(output_file, index=False, mode='a', header=False)

def print_update(epoch, batch_idx, training_loss, validation_loss):
    # print message
    tqdm.write("*" * 50)
    tqdm.write("epoch {} -- step {} -- loss: {:5.2f} -- ".\
               format(epoch, batch_idx, training_loss) + \
               "validation loss: {:5.2f}".\
               format(validation_loss))
