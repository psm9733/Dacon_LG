import tensorflow as tf
from network.backbone.regnet.regnet import RegNetY
from network.neck.neck import FPN
from network.common.layers import Conv2dBnAct
from network.common.blocks import StemBlock

def Classification_Head(in_tensor, num_classes, weight_decay, name):
    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    out_tensor = tf.keras.layers.Dense(units=1280, kernel_regularizer=kernel_regularizer)(in_tensor)
    out_tensor = tf.keras.layers.Dropout(0.2)(out_tensor)
    out_tensor = tf.keras.layers.Dense(units=num_classes, kernel_regularizer=kernel_regularizer, activation='softmax', name="Head_{}".format(name))(out_tensor)
    return out_tensor


def MultiScale_Classification_HEAD(in_tensor_list, activation, num_classes, weight_decay, name):
    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    out_tensor_list = []
    for index, in_tensor in enumerate(in_tensor_list):
        b, w, h, c = in_tensor.shape
        out_tensor = Conv2dBnAct(in_tensor, int(c), (3, 3), (1, 1), activation=activation, weight_decay=weight_decay)
        out_tensor = Conv2dBnAct(out_tensor, int(c / 2), (3, 3), (1, 1), activation=activation, weight_decay=weight_decay)
        out_tensor = tf.keras.layers.GlobalAveragePooling2D()(out_tensor)
        out_tensor = tf.keras.layers.Reshape((1, 1, -1))(out_tensor)
        b, w, h, c = out_tensor.shape
        out_tensor = tf.keras.layers.Conv2D(filters=c, activation=activation, kernel_size=(1, 1), strides=(1, 1),
                                            kernel_regularizer=kernel_regularizer)(out_tensor)
        out_tensor_list.append(out_tensor)
    out_tensor = tf.keras.layers.Concatenate()(out_tensor_list)
    out_tensor = tf.keras.layers.Dropout(0.2)(out_tensor)
    out_tensor = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=(1, 1), strides=(1, 1),
                                        kernel_regularizer=kernel_regularizer, activation='softmax')(out_tensor)
    out_tensor = tf.keras.layers.Reshape((-1,), name="Head_{}".format(name))(out_tensor)
    return out_tensor

def MultiScale_Regression_HEAD(in_tensor_list, activation, num_classes, weight_decay, name):
    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    out_tensor_list = []
    for index, in_tensor in enumerate(in_tensor_list):
        out_tensor = tf.keras.layers.GlobalAveragePooling2D()(in_tensor)
        out_tensor = tf.keras.layers.Reshape((1, 1, -1))(out_tensor)
        out_tensor = tf.keras.layers.Conv2D(filters=out_tensor.shape[-1], activation=activation, kernel_size=(1, 1), strides=(1, 1),
                                            kernel_regularizer=kernel_regularizer)(out_tensor)
        out_tensor_list.append(out_tensor)
    out_tensor = tf.keras.layers.Concatenate()(out_tensor_list)
    out_tensor = tf.keras.layers.Dropout(0.2)(out_tensor)
    out_tensor = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=(1, 1), strides=(1, 1),
                                        kernel_regularizer=kernel_regularizer, activation='linear')(out_tensor)
    out_tensor = tf.keras.layers.Reshape((-1,), name="Head_{}".format(name))(out_tensor)
    return out_tensor
