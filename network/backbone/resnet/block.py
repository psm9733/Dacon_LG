import tensorflow as tf
from tensorflow.keras.layers import *
from network.common.layers import Conv2dBnAct


def ResidualBlock(in_tensor, filter_list, kernel_size, strides, activation, weight_decay):
    '''
    :param in_tensor: Input tensor
    :param filter_list: Conv2d filter number list
    :param kernel_size: Conv2d filter size
    :param strides: Conv2d strides
    :param activation: Activation function
    :param weight_decay: weight_decay
    :return: Tensor
    '''
    conv2d_1_out_tensor = Conv2dBnAct(in_tensor, filter_list[0], (1, 1), (1, 1), activation=activation,
                                      weight_decay=weight_decay)
    conv2d_2_out_tensor = Conv2dBnAct(conv2d_1_out_tensor, filter_list[1], kernel_size, strides, activation=activation,
                                      weight_decay=weight_decay)
    conv2d_3_out_tensor = Conv2dBnAct(conv2d_2_out_tensor, filter_list[2], (1, 1), (1, 1), activation=activation,
                                      weight_decay=weight_decay)
    if conv2d_3_out_tensor.shape[1:] == in_tensor.shape[1:]:
        identity = in_tensor
    else:
        identity = Conv2dBnAct(in_tensor, filter_list[2], (1, 1), strides, activation=activation,
                               weight_decay=weight_decay)
    out_tensor = Add()([conv2d_3_out_tensor, identity])
    return out_tensor


if __name__ == "__main__":
    weight_decay = 1e-5
    input_shape = (416, 416, 3)
    input_tensor = tf.keras.layers.Input(input_shape)
    model = ResidualBlock(input_tensor, [32, 16, 32], (3, 3), (1, 1), "relu", weight_decay)
    model = tf.keras.Model(inputs=[input_tensor], outputs=model)
    model.summary()
    model.save("ResidualBlock.h5")
