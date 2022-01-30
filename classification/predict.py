import tensorflow as tf
import tensorflow_addons as tfa
import datetime
import os
import albumentations
import sys
import glob
import tqdm
import cv2
import numpy as np
from classification.config import *
from network.common.blocks import StemBlock
from network.backbone.regnet.regnet import RegNetY, RegNetZ
from network.neck.neck import FPN
from network.head.head import MultiScale_Classification_HEAD, MultiScale_Regression_HEAD
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
    model.load_weights("C:/Users/sangmin/Desktop/Dacon_LG/classification/saved_model/20220131-000947/RegNet_lr=0.001_wd=1e-05_batchsize=32_epoch=00160.h5")
    model.summary()

    img_list = glob.glob("../dataset/data/test/**/**.jpg", recursive=True)
    with open("submission_3.csv", 'w') as file:
        file.write("image,label\n")
        for img_path in tqdm.tqdm(img_list):
            img_path = img_path.replace("\\", "/")
            filename = os.path.basename(img_path).replace(".jpg", "")
            img = cv2.imread(img_path)
            resize_img = cv2.resize(img, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
            resize_img = np.expand_dims(resize_img.astype(np.float) / 255., axis=0)
            area, crop, disease, risk, total = model.predict(resize_img)
            crop_result = np.argmax(crop) + 1
            disease_value = np.argmax(disease)
            disease_result = [key for key, value in disease_dict.items() if value == disease_value][0]
            risk_result = np.argmax(risk)
            total_value = np.argmax(total)
            total_result = [key for key, value in label_description.items() if value == total_value][0]
            # print(filename)
            # print(total_result)
            file.write(filename + "," + total_result + "\n")


