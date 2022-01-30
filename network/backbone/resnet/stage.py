import tensorflow as tf
from network.backbone.resnet.block import ResidualBlock


def ResidualStage(in_tensor, n_block, filter_list, kernel_size_list, strides, activation, weight_decay):
    '''
    :param in_tensor: Input tensor
    :param n_block: block depth of this stage
    :param filter_list: this stage filter number
    :param kernel_size_list: this stage kernel size
    :param strides: this stage strides
    :param activation: Activation function
    :param weight_decay: weight_decay
    :return: Tensor
    '''
    out_tensor = in_tensor
    for index in range(n_block):
        if index == 0:
            out_tensor = ResidualBlock(out_tensor, filter_list, kernel_size_list, strides, activation, weight_decay)
        else:
            out_tensor = ResidualBlock(out_tensor, filter_list, kernel_size_list, (1, 1), activation, weight_decay)
    return out_tensor


if __name__ == "__main__":
    weight_decay = 1e-5
    input_shape = (416, 416, 3)
    input_tensor = tf.keras.layers.Input(input_shape)
    model = ResidualStage(input_tensor, 3, [32, 16, 32], (3, 3), (2, 2), "relu", weight_decay)
    model = tf.keras.Model(inputs=[input_tensor], outputs=model)
    model.summary()
    model.save("ResidualStage.h5")
