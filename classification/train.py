import tensorflow as tf
import tensorflow_addons as tfa
import datetime
import os
import albumentations
import sys
from classification.config import *
from network.common.blocks import StemBlock
from network.backbone.regnet.regnet import RegNetY, RegNetZ
from network.neck.neck import FPN
from network.head.head import Classification_Head, MultiScale_Classification_HEAD, MultiScale_Regression_HEAD
from classification.generator import MultiTask_Generator
from utils.logger import Logger
from utils.scheduler import CosineAnnealingLRScheduler
from tensorflow_addons.optimizers import RectifiedAdam

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def Model(in_shape, num_classes_list, n_block_per_stage, filter_per_stage, kernel_size_per_stage,
          strides_per_stage, groups_per_stage, activation,
          weight_decay):
    input_tensor = tf.keras.layers.Input(in_shape)
    stem = StemBlock(input_tensor, [32, 16, 32], [(3, 3), (3, 3), (1, 1)], [(2, 2), (2, 2), (1, 1)], activation,
                     weight_decay)
    backbone = RegNetZ(
        stem,
        n_block_per_stage,
        filter_per_stage,
        kernel_size_per_stage,
        strides_per_stage,
        groups_per_stage,
        activation,
        weight_decay
    )

    fpn_area = FPN(backbone, activation, weight_decay, "add")
    head_area = MultiScale_Classification_HEAD(fpn_area, activation, num_classes_list[0], weight_decay, "area")

    fpn_crop = FPN(backbone, activation, weight_decay, "add")
    head_crop = MultiScale_Classification_HEAD(fpn_crop, activation, num_classes_list[1], weight_decay, "crop")

    fpn_disease = FPN(backbone, activation, weight_decay, "add")
    head_disease = MultiScale_Classification_HEAD(fpn_disease, activation, num_classes_list[2], weight_decay, "disease")

    fpn_risk = FPN(backbone, activation, weight_decay, "add")
    head_risk = MultiScale_Classification_HEAD(fpn_risk, activation, num_classes_list[3], weight_decay, "risk")

    fpn_total = FPN(backbone, activation, weight_decay, "add")
    head_total = MultiScale_Classification_HEAD(fpn_total, activation, num_classes_list[4], weight_decay, "total")

    model = tf.keras.Model(inputs=[input_tensor], outputs=[head_area, head_crop, head_disease, head_risk, head_total])
    return model


if __name__ == '__main__':
    backbone_name = "RegNet"
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
    model = Model(INPUT_SHAPE,
                            num_classes_list,
                            [1, 1, 1],
                            [[128, 128, 128], [256, 256, 256], [512, 512, 512]],
                            [(3, 3), (3, 3), (3, 3)],
                            [(2, 2), (2, 2), (2, 2)],
                            [64, 64, 64],
                            activation,
                            WEIGHT_DECAY)
    # model.load_weights(
    #     "C:/Users/sangmin/Desktop/Dacon_LG/classification/saved_model/20220130-161005/RegNet_lr=0.001_wd=1e-05_batchsize=32_epoch=00110.h5")
    model.summary()
    # model.save("multi_task.h5")
    for output in model.outputs:
        print("output tensor = ", output.get_shape())

    train_transform = albumentations.Compose([
        albumentations.Resize(height=INPUT_SHAPE[0], width=INPUT_SHAPE[1]),
        albumentations.SomeOf([
            albumentations.RandomRotate90(p=1),
            albumentations.Sharpen(),
        ], 2, p=0.5),
        albumentations.SomeOf([
            albumentations.RandomBrightness(),
            albumentations.Affine(),
            albumentations.RandomContrast(),
            albumentations.Solarize(),
            albumentations.ColorJitter(),
        ], 3, p=0.5),
        albumentations.Flip(p=0.5),
    ])

    valid_transform = albumentations.Compose([
        albumentations.Resize(height=INPUT_SHAPE[0], width=INPUT_SHAPE[1]),
    ])

    train_batch_gen = MultiTask_Generator(dataset_info_path=TRAIN_PATH, batch_size=BATCH_SIZE,
                                         input_shape=INPUT_SHAPE,
                                         num_classes_list=num_classes_list, augs=train_transform, is_train=True)
    valid_batch_gen = MultiTask_Generator(dataset_info_path=VALID_PATH, batch_size=max(1, BATCH_SIZE),
                                         input_shape=INPUT_SHAPE,
                                         num_classes_list=num_classes_list, augs=valid_transform, is_train=False)

    model.compile(
        optimizer=RectifiedAdam(LR),
        loss={
            "Head_area": tf.keras.losses.categorical_crossentropy,
            "Head_crop": tf.keras.losses.categorical_crossentropy,
            "Head_disease": tf.keras.losses.categorical_crossentropy,
            "Head_risk": tf.keras.losses.categorical_crossentropy,
            "Head_total": tf.keras.losses.categorical_crossentropy,
        },
        metrics={
            "Head_area": tfa.metrics.F1Score(num_classes=num_classes_list[0]),
            "Head_crop": tfa.metrics.F1Score(num_classes=num_classes_list[1]),
            "Head_disease": tfa.metrics.F1Score(num_classes=num_classes_list[2]),
            "Head_risk": tfa.metrics.F1Score(num_classes=num_classes_list[3]),
            "Head_total": tfa.metrics.F1Score(num_classes=num_classes_list[4]),
        }
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=save_dir + '/' + model_name + '_epoch={epoch:05d}.h5',
                                                    monitor='val_loss',
                                                    save_best_only=True,
                                                    save_weights_only=False,
                                                    verbose=1,
                                                    period=1),
        tf.keras.callbacks.TensorBoard(log_dir),
        CosineAnnealingLRScheduler(total_epochs=EPOCHS, init_lr=LR, warmup_epoch=WARMUP_EPOCHS, n_cycles=N_CYCLES,
                                   lr_decay_rate=LR_DECAY_LATE, verbose=True, logger=lr_logger)
    ]

    model.fit_generator(train_batch_gen,
                        use_multiprocessing=False,
                        max_queue_size=20,
                        callbacks=callbacks,
                        workers=1,
                        epochs=EPOCHS,
                        validation_data=valid_batch_gen
)
