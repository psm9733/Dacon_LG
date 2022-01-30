import tensorflow as tf
from network.backbone.resnet.stage import ResidualStage


def ResNetBody(in_tensor, n_block_per_stage, filter_per_stage, kernel_size_per_stage, strides_per_stage, activation,
               weight_decay):
    '''
    :param in_tensor: Input tensor
    :param n_block_per_stage: block depth per stage
    :param filter_per_stage: filter number per stage
    :param kernel_size_per_stage: kernel size per stage
    :param strides_per_stage: strides per stage
    :param activation: Activation function
    :param weight_decay: weight_decay
    :return: Tensor
    '''
    n_stage = len(n_block_per_stage)
    out_tensor = in_tensor
    out_tensor_list = []
    for index in range(n_stage):
        out_tensor = ResidualStage(out_tensor, n_block_per_stage[index], filter_per_stage[index],
                                   kernel_size_per_stage[index], strides_per_stage[index], activation, weight_decay)
        out_tensor_list.append(out_tensor)
    return out_tensor_list


if __name__ == "__main__":
    weight_decay = 1e-5
    input_shape = (416, 416, 3)
    input_tensor = tf.keras.layers.Input(input_shape)
    model = ResNetBody(input_tensor, [1, 1, 1, 1], [[32, 32, 32], [64, 64, 64], [128, 128, 128], [256, 256, 256]],
                       [(3, 3), (3, 3), (3, 3), (3, 3)], [(2, 2), (2, 2), (2, 2), (2, 2)], "relu", weight_decay)
    model = tf.keras.Model(inputs=[input_tensor], outputs=model)
    model.summary()
    model.save("ResNetBody.h5")
