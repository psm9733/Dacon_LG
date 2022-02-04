import tensorflow as tf
import tensorflow_addons as tfa
import datetime
import pandas as pd
import os
import albumentations
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import glob
import tqdm
import natsort.natsort
import cv2
import numpy as np
from classification.config import *
from network.common.blocks import StemBlock
from network.backbone.regnet.regnet import RegNetY, RegNetZ
from network.neck.neck import FPN
from network.head.head import MultiScale_Classification_HEAD, MultiScale_Regression_HEAD, MultiScale_Segmentation_HEAD
from classification.generator import MultiTask_Generator
from utils.logger import Logger
from utils.scheduler import CosineAnnealingLRScheduler
from tensorflow_addons.optimizers import RectifiedAdam

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def Model(cnn_shape, embedding_length, num_classes_list, n_block_per_stage, filter_per_stage, kernel_size_per_stage,
          strides_per_stage, groups_per_stage, activation,
          weight_decay):
    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    input_tensor = tf.keras.layers.Input(cnn_shape)
    embed_tensor = tf.keras.layers.Input(shape=(embedding_length))
    stem = StemBlock(input_tensor, [64, 32, 64], [(3, 3), (3, 3), (1, 1)], [(2, 2), (2, 2), (1, 1)], activation,
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

    fpn = FPN(backbone, activation, weight_decay, "add")
    fpn_feature_0 = tf.keras.layers.GlobalAveragePooling2D()(fpn[0])
    fpn_feature_1 = tf.keras.layers.GlobalAveragePooling2D()(fpn[1])
    fpn_feature_2 = tf.keras.layers.GlobalAveragePooling2D()(fpn[2])
    fpn_features = tf.keras.layers.Concatenate()([fpn_feature_0, fpn_feature_1, fpn_feature_2])
    # head_seg_0, head_seg_1, head_seg_2 = MultiScale_Segmentation_HEAD(fpn, activation, weight_decay, "seg")

    head_area = MultiScale_Classification_HEAD(fpn, activation, num_classes_list[0], weight_decay, "area")

    head_crop = MultiScale_Classification_HEAD(fpn, activation, num_classes_list[1], weight_decay, "crop")

    head_disease = MultiScale_Classification_HEAD(fpn, activation, num_classes_list[2], weight_decay, "disease")

    head_risk = MultiScale_Classification_HEAD(fpn, activation, num_classes_list[3], weight_decay, "risk")

    cnn_concat = tf.keras.layers.Concatenate()([head_area, head_crop, head_disease, head_risk])
    cnn_concat = tf.keras.layers.Dropout(0.1)(cnn_concat)
    cnn_concat = tf.keras.layers.Dense(int(fpn_features.shape[-1] / 2), activation=activation)(cnn_concat)

    # head_env = tf.keras.layers.Dense(embed_tensor.shape[-1])(embed_tensor)
    head_env = tf.keras.layers.Reshape((1, -1))(embed_tensor)

    lstm_tensor = tf.keras.layers.LSTM(head_env.shape[-1], activation=activation, return_sequences=True,
                                       kernel_regularizer=kernel_regularizer)(head_env)
    lstm_tensor = tf.keras.layers.Dropout(0.1)(lstm_tensor)
    lstm_tensor = tf.keras.layers.Dense(int(fpn_features.shape[-1] / 2), activation=activation)(lstm_tensor)
    lstm_tensor = tf.keras.layers.Reshape((-1, ))(lstm_tensor)

    total_concat_tensor = tf.keras.layers.Concatenate()([fpn_features, cnn_concat, lstm_tensor])
    head_total = tf.keras.layers.Dropout(0.1)(total_concat_tensor)
    head_total = tf.keras.layers.Dense(num_classes_list[4], activation="softmax", kernel_regularizer=kernel_regularizer,
                                       name="Head_total")(head_total)
    # head_total = MultiScale_Classification_HEAD(fpn, activation, num_classes_list[4], weight_decay, "total")
    model = tf.keras.Model(inputs=[input_tensor, embed_tensor],
                           outputs=[head_area, head_crop, head_disease, head_risk, head_total])
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
    model = Model(INPUT_SHAPE, EMBEDDING_LENGTH,
                            num_classes_list,
                            [1, 1, 1],
                            [[128, 128, 128], [256, 256, 256], [512, 512, 512]],
                            [(3, 3), (3, 3), (3, 3)],
                            [(2, 2), (2, 2), (2, 2)],
                            [64, 64, 64],
                            activation,
                            WEIGHT_DECAY)
    model.load_weights("/home/fssv1/sangmin/Dacon_LG/classification/saved_model/20220204-153027/RegNet_lr=0.0005_wd=1e-05_batchsize=64_epoch=00097.h5")
    model.summary()
    csv_features = ['측정시각', '내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 습도 1 평균', '내부 습도 1 최고',
                         '내부 습도 1 최저', '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저']
    img_list = glob.glob("../dataset/data/train/**/**.jpg", recursive=True)
    temp_csv = pd.read_csv(img_list[0].replace(".jpg", ".csv"))[csv_features[1:]]
    max_arr, min_arr = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()
    for img_path in tqdm.tqdm(img_list[1:]):
        csv = img_path.replace(".jpg", ".csv")
        temp_csv = pd.read_csv(csv)[csv_features[1:]]
        temp_csv = temp_csv.replace('-', np.nan).dropna()
        if len(temp_csv) == 0:
            continue
        temp_csv = temp_csv.astype(float)
        temp_max, temp_min = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()
        max_arr = np.max([max_arr, temp_max], axis=0)
        min_arr = np.min([min_arr, temp_min], axis=0)

    # feature 별 최대값, 최솟값 dictionary 생성
    csv_feature_dict = {csv_features[i + 1]: [min_arr[i], max_arr[i]] for i in
                             range(len(csv_features[1:]))}
    img_list = natsort.natsorted(glob.glob("../dataset/data/test/**/**.jpg", recursive=True))
    with open("submission_9.csv", 'w') as file:
        file.write("image,label\n")
        for img_path in tqdm.tqdm(img_list):
            img_path = img_path.replace("\\", "/")
            csv_path = img_path.replace(".jpg", ".csv")
            csv = pd.read_csv(csv_path)[csv_features]
            csv = csv.replace('-', 0)
            for col in csv.columns[1:]:
                csv[col] = csv[col].astype(float) - csv_feature_dict[col][0]
                csv[col] = csv[col] / (csv_feature_dict[col][1] - csv_feature_dict[col][0])
            if len(csv) == 0:
                continue
            csv_max = csv.max().to_numpy()
            csv_min = csv.min().to_numpy()
            csv_max[0] = pd.to_datetime(csv_max[0]).month / 12
            csv_min[0] = pd.to_datetime(csv_min[0]).month / 12
            gt = np.concatenate((csv_min, csv_max))
            gt = np.expand_dims(gt.astype(np.float), axis=0)

            filename = os.path.basename(img_path).replace(".jpg", "")
            img = cv2.imread(img_path)
            resize_img = cv2.resize(img, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
            resize_img = np.expand_dims(resize_img.astype(np.float) / 255., axis=0)
            area, crop, disease, risk, total = model.predict([resize_img, gt])
            crop_result = np.argmax(crop) + 1
            disease_value = np.argmax(disease)
            disease_result = [key for key, value in disease_dict.items() if value == disease_value][0]
            risk_result = np.argmax(risk)
            total_value = np.argmax(total)
            total_result = [key for key, value in label_description.items() if value == total_value][0]
            # print(filename)
            # print(total_result)
            file.write(filename + "," + total_result + "\n")


