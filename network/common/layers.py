import tensorflow as tf
from tensorflow.keras.layers import *


def Conv2dBnAct(in_tensor, filters, kernel_size, strides, padding="same", groups=1, activation=None, use_bias=False,
                weight_decay=1e-5, bn_momentum=0.9):
    '''
    :param in_tensor: Input tensor
    :param filters: Conv2d filter number
    :param kernel_size: Conv2d filter size
    :param strides: Conv2d strides
    :param padding: Conv2d padding
    :param groups: Conv2d groups(if groups is greater than 1, it is group conv2d)
    :param activation: Conv2d activation function
    :param use_bias: Conv2d bias
    :param weight_decay: weight_decay
    :param bn_momentum: BatchNormalization momentum
    :return: Tensor
    '''
    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    out_tensor = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, groups=groups,
                    use_bias=use_bias, kernel_regularizer=kernel_regularizer,
                    kernel_initializer=tf.keras.initializers.he_normal())(in_tensor)
    out_tensor = BatchNormalization(momentum=bn_momentum)(out_tensor)
    out_tensor = Activation(activation)(out_tensor)
    return out_tensor


if __name__ == "__main__":
    weight_decay = 5e-4
    input_shape = (416, 416, 3)
    input_tensor = tf.keras.layers.Input(input_shape)
    model = Conv2dBnAct(input_tensor, 32, (3, 3), 1, "same", 1, "relu")
    model = tf.keras.Model(inputs=[input_tensor], outputs=model)
    model.summary()
