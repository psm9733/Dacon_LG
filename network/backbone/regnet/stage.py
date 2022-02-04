import tensorflow as tf
from network.backbone.regnet.block import XBlock, YBlock, ZBlock


def RegNetXStage(in_tensor, n_block, filter_list, kernel_size_list, strides, groups, activation, weight_decay):
    '''
    :param in_tensor: Input tensor
    :param n_block: block  of this stage
    :param filter_list: this stage filter number
    :param kernel_size_list: this stage kernel size
    :param strides: this stage strides
    :param groups: Conv2d groups option
    :param activation: Activation function
    :param weight_decay: weight_decay
    :return: Tensor
    '''
    out_tensor = in_tensor
    for index in range(n_block):
        if index == 0:
            out_tensor = XBlock(out_tensor, filter_list, kernel_size_list, strides, groups, activation, weight_decay)
        else:
            out_tensor = XBlock(out_tensor, filter_list, kernel_size_list, (1, 1), groups, activation, weight_decay)
    return out_tensor


def RegNetYStage(in_tensor, n_block, filter_list, kernel_size_list, strides, groups, activation, weight_decay, se_reduction_ratio):
    '''
    :param in_tensor: Input tensor
    :param n_block: block  of this stage
    :param filter_list: this stage filter number
    :param kernel_size_list: this stage kernel size
    :param strides: this stage strides
    :param groups: Conv2d groups option
    :param activation: Activation function
    :param weight_decay: weight_decay
    :param se_reduction_ratio: SeBlock reduction ratio
    :return: Tensor
    '''
    out_tensor = in_tensor
    for index in range(n_block):
        if index == 0:
            out_tensor = YBlock(out_tensor, filter_list, kernel_size_list, strides, groups, activation, weight_decay, se_reduction_ratio)
        else:
            out_tensor = YBlock(out_tensor, filter_list, kernel_size_list, (1, 1), groups, activation, weight_decay, se_reduction_ratio)
    return out_tensor

def RegNetZStage(in_tensor, n_block, filter_list, kernel_size_list, strides, groups, activation, weight_decay, se_reduction_ratio):
    '''
    :param in_tensor: Input tensor
    :param n_block: block  of this stage
    :param filter_list: this stage filter number
    :param kernel_size_list: this stage kernel size
    :param strides: this stage strides
    :param groups: Conv2d groups option
    :param activation: Activation function
    :param weight_decay: weight_decay
    :param cb_reduction_ratio: SeBlock reduction ratio
    :return: Tensor
    '''
    out_tensor = in_tensor
    for index in range(n_block):
        if index == 0:
            out_tensor = ZBlock(out_tensor, filter_list, kernel_size_list, strides, groups, activation, weight_decay, se_reduction_ratio)
        else:
            out_tensor = ZBlock(out_tensor, filter_list, kernel_size_list, (1, 1), groups, activation, weight_decay, se_reduction_ratio)
    return out_tensor

if __name__ == "__main__":
    weight_decay = 1e-5
    input_shape = (416, 416, 3)
    input_tensor = tf.keras.layers.Input(input_shape)
    model = RegNetYStage(input_tensor, 3, [32, 16, 32], (3, 3), (2, 2), 16, "relu", weight_decay, 16)
    model = tf.keras.Model(inputs=[input_tensor], outputs=model)
    model.summary()
    # model.save("RegNetXStage.h5")
