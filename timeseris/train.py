import tensorflow as tf
import datetime
import os
from model import TimeSerisModel
from config import *
from utils.logger import Logger
from generator import TimeSerisGenerator
from utils.scheduler import CosineAnnealingLRScheduler
from tensorflow_addons.optimizers import RectifiedAdam

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    backbone_name = "ResNet"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "./logs/fit/" + timestamp
    model_name = backbone_name + "_lr=" + str(LR) + "_wd=" + str(WEIGHT_DECAY) + "_batchsize=" + str(
        BATCH_SIZE)
    save_dir = "./saved_model/" + timestamp
    if os.path.isdir('./logs') == False:
        os.mkdir('./logs')
    if os.path.isdir('./logs/fit') == False:
        os.mkdir('./logs/fit')
    if os.path.isdir('./saved_model') == False:
        os.mkdir('./saved_model')
    if os.path.isdir(save_dir) == False:
        os.mkdir(save_dir)
    if os.path.isdir(log_dir) == False:
        os.mkdir(log_dir)

    lr_logger = Logger(log_dir + "/lr/" + model_name)
    activation = tf.keras.layers.Activation(tf.nn.relu)
    model = TimeSerisModel(INPUT_SHAPE, NUM_CLASSES, 4, activation, WEIGHT_DECAY)
    output_shape_list = []
    for output in model.outputs:
        output_shape_list.append(output.get_shape()[1:3])
        print("output tensor = " + output.name + str(output.get_shape()))

    train_batch_gen = TimeSerisGenerator(dataset_dir_path=TRAIN_DIR_PATH, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE,
                                num_classes=NUM_CLASSES, is_train=True)
    valid_batch_gen = TimeSerisGenerator(dataset_dir_path=VALID_DIR_PATH, batch_size=max(1, int(BATCH_SIZE / 8)),
                                input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
                                is_train=False)

    model.compile(
        optimizer=RectifiedAdam(LR),
        loss=tf.losses.MSE(),
    )

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=save_dir + '/' + model_name + '_epoch={epoch:05d}.h5',
                                                    monitor='val_loss',
                                                    save_best_only=True,
                                                    save_weights_only=False,
                                                    verbose=1,
                                                    period=1),
                 tf.keras.callbacks.TensorBoard(log_dir),
                 CosineAnnealingLRScheduler(total_epochs=EPOCHS, init_lr=LR, warmup_epoch=WARMUP_EPOCHS, n_cycles=4,
                                            lr_decay_rate=LR_DECAY_LATE, verbose=True, logger=lr_logger)
                 ]

    model.fit_generator(train_batch_gen,
                        use_multiprocessing=True,
                        max_queue_size=20,
                        callbacks=callbacks,
                        workers=4,
                        epochs=EPOCHS,
                        validation_data=valid_batch_gen)
