import tensorflow as tf
from tensorflow.keras.layers import *
from network.common.layers import Conv2dBnAct


def StemBlock(in_tensor, filter_list, kernel_size_list, strides_list, activation, weight_decay):
    '''
    :param in_tensor: Input tensor
    :param filter_list: Conv2d filter number list
    :param kernel_size: Conv2d filter size list
    :param strides_list: Conv2d strides list
    :param activation: Activation function
    :param weight_decay: weight_decay
    :return: Tensor
    '''
    conv2d_1_out_tensor = Conv2dBnAct(in_tensor, filter_list[0], kernel_size_list[0], strides_list[0],
                                      activation=activation,
                                      weight_decay=weight_decay)

    conv2d_2_1_out_tensor = Conv2dBnAct(conv2d_1_out_tensor, filter_list[1] / 2, (1, 1), (1, 1), activation=activation,
                                        weight_decay=weight_decay)
    conv2d_2_2_out_tensor = Conv2dBnAct(conv2d_2_1_out_tensor, filter_list[1], kernel_size_list[1], strides_list[1],
                                        activation=activation, weight_decay=weight_decay)

    identity_out_tensor = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2d_1_out_tensor)

    concat_out_tensor = Concatenate()([conv2d_2_2_out_tensor, identity_out_tensor])
    conv2d_3_out_tensor = Conv2dBnAct(concat_out_tensor, filter_list[2], kernel_size_list[2], strides_list[2],
                                      activation=activation, weight_decay=weight_decay)
    return conv2d_3_out_tensor


def SEBlock(in_tensor, activation, weight_decay, reduction_ratio=4):
    '''
    :param in_tensor: Input tensor
    :param reduction_ratio: Dimensionlity-reduction - reduction ratio(r) value
    :param weight_decay: weight_decay
    :return: Tensor
    '''
    out_tensor = tf.keras.layers.GlobalAveragePooling2D()(in_tensor)
    out_tensor = tf.keras.layers.Reshape((1, 1, -1))(out_tensor)
    out_filters = out_tensor.shape[-1]
    out_tensor = Conv2dBnAct(out_tensor, max(1, int(out_filters / reduction_ratio)), (1, 1), (1, 1),
                             activation=activation, weight_decay=weight_decay)
    out_tensor = Conv2dBnAct(out_tensor, out_filters, (1, 1), (1, 1),
                             activation="sigmoid", weight_decay=weight_decay)
    out_tensor = in_tensor * out_tensor
    return out_tensor


def CbamBlock(cbam_feature, ratio=4):
    cbam_feature = ChannelAttention(cbam_feature, ratio)
    cbam_feature = SpatialAttention(cbam_feature)
    return cbam_feature


def ChannelAttention(input_feature, ratio=4):
    channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = tf.keras.layers.Dense(channel // ratio,
                                          activation='relu',
                                          kernel_initializer='he_normal',
                                          use_bias=True,
                                          bias_initializer='zeros')
    shared_layer_two = tf.keras.layers.Dense(channel,
                                          kernel_initializer='he_normal',
                                          use_bias=True,
                                          bias_initializer='zeros')

    avg_pool = tf.keras.layers.AveragePooling2D(pool_size=input_feature.shape[1],
                                             strides=input_feature.shape[1])(input_feature)
    avg_pool = tf.keras.layers.Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)

    max_pool = tf.keras.layers.MaxPooling2D(pool_size=input_feature.shape[1],
                                         strides=input_feature.shape[1])(input_feature)
    max_pool = tf.keras.layers.Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)

    cbam_feature = tf.keras.layers.Add()([avg_pool, max_pool])
    cbam_feature = tf.keras.layers.Activation('sigmoid')(cbam_feature)

    if tf.keras.backend.image_data_format() == "channels_first":
        cbam_feature = tf.keras.layers.Permute((3, 1, 2))(cbam_feature)

    return tf.keras.layers.multiply([input_feature, cbam_feature])

def SpatialAttention(input_feature):
    kernel_size = 7

    if tf.keras.backend.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = tf.keras.layers.Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool1 = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(cbam_feature)
    avg_pool2 = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(cbam_feature)
    concat = tf.keras.layers.Concatenate(axis=3)([avg_pool1, avg_pool2])

    cbam_feature = tf.keras.layers.Conv2D(filters=1,
                                       kernel_size=kernel_size,
                                       strides=1,
                                       padding='same',
                                       activation='sigmoid',
                                       kernel_initializer='he_normal',
                                       use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1

    if tf.keras.backend.image_data_format == "channels_first":
        cbam_feature = tf.keras.layers.Permute((3, 1, 2))(cbam_feature)

    return tf.keras.layers.multiply([input_feature, cbam_feature])

if __name__ == "__main__":
    weight_decay = 1e-5
    input_shape = (416, 416, 3)
    input_tensor = tf.keras.layers.Input(input_shape)
    model = StemBlock(input_tensor, [32, 16, 32], [(3, 3), (3, 3), (1, 1)], [(2, 2), (2, 2), (1, 1)], "relu",
                      weight_decay)
    se_block = SEBlock(model, "relu", weight_decay, 2)
    se_block = tf.keras.Model(inputs=[input_tensor], outputs=se_block)
    se_block.summary()
    se_block.save("se_block.h5")
