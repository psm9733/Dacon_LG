import tensorflow as tf
from network.backbone.regnet.stage import RegNetXStage, RegNetYStage, RegNetZStage


def RegNetXBody(in_tensor, n_block_per_stage, filter_per_stage, kernel_size_per_stage, strides_per_stage,
                groups_per_stage, activation, weight_decay):
    '''
    :param in_tensor: Input tensor
    :param n_block_per_stage: block depth per stage
    :param filter_per_stage: filter number per stage
    :param bottleneck_ratio_per_stage: bottleneck ratio per stage
    :param kernel_size_per_stage: kernel size per stage
    :param strides_per_stage: strides per stage
    :param groups_per_stage: groups per stage
    :param activation: Activation function
    :param weight_decay: weight_decay
    :return: Tensor
    '''
    n_stage = len(n_block_per_stage)
    out_tensor = in_tensor
    out_tensor_list = []
    for index in range(n_stage):
        out_tensor = RegNetXStage(out_tensor, n_block_per_stage[index], filter_per_stage[index],
                                  kernel_size_per_stage[index], strides_per_stage[index], groups_per_stage[index],
                                  activation, weight_decay)
        out_tensor_list.append(out_tensor)
    return out_tensor_list


def RegNetYBody(in_tensor, n_block_per_stage, filter_per_stage, kernel_size_per_stage, strides_per_stage,
                groups_per_stage, activation, weight_decay, se_reduction_ratio):
    '''
    :param in_tensor: Input tensor
    :param n_block_per_stage: block depth per stage
    :param filter_per_stage: filter number per stage
    :param bottleneck_ratio_per_stage: bottleneck ratio per stage
    :param kernel_size_per_stage: kernel size per stage
    :param strides_per_stage: strides per stage
    :param groups_per_stage: groups per stage
    :param activation: Activation function
    :param weight_decay: weight_decay
    :param se_reduction_ratio_per_stage: SE Block reduction ratio per stage
    :return: Tensor
    '''
    n_stage = len(n_block_per_stage)
    out_tensor = in_tensor
    out_tensor_list = []
    for index in range(n_stage):
        out_tensor = RegNetYStage(out_tensor, n_block_per_stage[index], filter_per_stage[index],
                                  kernel_size_per_stage[index], strides_per_stage[index], groups_per_stage[index],
                                  activation, weight_decay, se_reduction_ratio[index])
        out_tensor_list.append(out_tensor)
    return out_tensor_list

def RegNetZBody(in_tensor, n_block_per_stage, filter_per_stage, kernel_size_per_stage, strides_per_stage,
                groups_per_stage, activation, weight_decay, cb_reduction_ratio):
    '''
    :param in_tensor: Input tensor
    :param n_block_per_stage: block depth per stage
    :param filter_per_stage: filter number per stage
    :param bottleneck_ratio_per_stage: bottleneck ratio per stage
    :param kernel_size_per_stage: kernel size per stage
    :param strides_per_stage: strides per stage
    :param groups_per_stage: groups per stage
    :param activation: Activation function
    :param weight_decay: weight_decay
    :param cb_reduction_ratio_per_stage: SE Block reduction ratio per stage
    :return: Tensor
    '''
    n_stage = len(n_block_per_stage)
    out_tensor = in_tensor
    out_tensor_list = []
    for index in range(n_stage):
        out_tensor = RegNetZStage(out_tensor, n_block_per_stage[index], filter_per_stage[index],
                                  kernel_size_per_stage[index], strides_per_stage[index], groups_per_stage[index],
                                  activation, weight_decay, cb_reduction_ratio[index])
        out_tensor_list.append(out_tensor)
    return out_tensor_list

if __name__ == "__main__":
    weight_decay = 1e-5
    input_shape = (416, 416, 3)
    input_tensor = tf.keras.layers.Input(input_shape)
    model = RegNetYBody(input_tensor,
                        [1, 1, 1, 1],
                        [[32, 32, 32], [64, 64, 64], [128, 128, 128], [256, 256, 256]],
                        [(3, 3), (3, 3), (3, 3), (3, 3)],
                        [(2, 2), (2, 2), (2, 2), (2, 2)],
                        [16, 16, 16, 16],
                        "relu",
                        weight_decay,
                        [4, 4, 4, 4])
    model = tf.keras.Model(inputs=[input_tensor], outputs=model)
    model.summary()
    # model.save("RegNetXBody.h5")
