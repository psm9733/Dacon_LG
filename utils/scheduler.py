import tensorflow as tf
import tensorflow.keras.backend as backend
import math

class CosineAnnealingLRScheduler(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, init_lr, warmup_epoch, n_cycles, logger, lr_decay_rate=0, verbose=False):
        self.total_epochs = total_epochs
        self.init_lr = init_lr
        self.warmup_epoch = warmup_epoch
        self.n_cycles = n_cycles
        self.logger = logger
        self.lr_decay_rate = lr_decay_rate
        self.verbose = verbose
        self.epoch_per_cycle = total_epochs / n_cycles
        self.lr_history = []

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epoch:
            remains = epoch % self.warmup_epoch
            lr = self.init_lr * math.sin((math.pi / 2) * remains / self.warmup_epoch)
        elif epoch >= self.warmup_epoch:
            cycles_index = (epoch // self.epoch_per_cycle)
            remains = (epoch % self.epoch_per_cycle)
            if remains == 0:
                self.init_lr = self.init_lr * math.pow((1 - self.lr_decay_rate), cycles_index)
            if cycles_index == 0:
                lr = self.init_lr * math.cos((math.pi / 2) * (remains - self.warmup_epoch) / (self.epoch_per_cycle - self.warmup_epoch))
            else:
                lr = self.init_lr * math.cos((math.pi / 2) * remains / self.epoch_per_cycle)
        backend.set_value(self.model.optimizer.lr, lr)
        self.lr_history.append(lr)
        self.logger.write('lr', data=backend.get_value(lr), step=epoch)
        if self.verbose:
            print("\nEpoch %05d, CosineAnnealingLRScheduler setting learning rate to %s" % (
                epoch + 1, backend.get_value(self.model.optimizer.lr)))