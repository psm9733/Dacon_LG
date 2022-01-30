import tensorflow as tf
from network.common.blocks import StemBlock
from network.backbone.regnet.body import RegNetXBody, RegNetYBody, RegNetZBody


def RegNetX(in_tensor, n_block_per_stage, filter_per_stage, kernel_size_per_stage, strides_per_stage, groups_per_stage,
            activation, weight_decay):
    '''
    :param in_tensor: Input tensor
    :param n_block_per_stage: block depth per stage
    :param filter_per_stage: filter number per stage
    :param kernel_size_per_stage: kernel size per stage
    :param strides_per_stage: strides per stage
    :param groups_per_stage: groups per stage
    :param activation: Activation function
    :param weight_decay: weight_decay
    :return: Tensor
    '''
    body_out = RegNetXBody(in_tensor, n_block_per_stage, filter_per_stage, kernel_size_per_stage, strides_per_stage,
                           groups_per_stage, activation, weight_decay)
    return body_out


def RegNetY(in_tensor, n_block_per_stage, filter_per_stage, kernel_size_per_stage, strides_per_stage, groups_per_stage,
            activation, weight_decay, se_reduction_ratio_per_stage=[2, 2, 2, 2]):
    '''
    :param in_tensor: Input tensor
    :param n_block_per_stage: block depth per stage
    :param filter_per_stage: filter number per stage
    :param kernel_size_per_stage: kernel size per stage
    :param strides_per_stage: strides per stage
    :param groups_per_stage: groups per stage
    :param activation: Activation function
    :param weight_decay: weight_decay
    :param se_reduction_ratio_per_stage: SE Block reduction ratio per stage
    :return: Tensor
    '''
    body_out = RegNetYBody(in_tensor, n_block_per_stage, filter_per_stage, kernel_size_per_stage, strides_per_stage,
                           groups_per_stage, activation, weight_decay, se_reduction_ratio_per_stage)
    return body_out

def RegNetZ(in_tensor, n_block_per_stage, filter_per_stage, kernel_size_per_stage, strides_per_stage, groups_per_stage,
            activation, weight_decay, cb_reduction_ratio_per_stage=[2, 2, 2, 2]):
    '''
    :param in_tensor: Input tensor
    :param n_block_per_stage: block depth per stage
    :param filter_per_stage: filter number per stage
    :param kernel_size_per_stage: kernel size per stage
    :param strides_per_stage: strides per stage
    :param groups_per_stage: groups per stage
    :param activation: Activation function
    :param weight_decay: weight_decay
    :param cb_reduction_ratio_per_stage: SE Block reduction ratio per stage
    :return: Tensor
    '''
    body_out = RegNetZBody(in_tensor, n_block_per_stage, filter_per_stage, kernel_size_per_stage, strides_per_stage,
                           groups_per_stage, activation, weight_decay, cb_reduction_ratio_per_stage)
    return body_out


if __name__ == "__main__":
    weight_decay = 1e-5
    input_shape = (64, 64, 3)
    input_tensor = tf.keras.layers.Input(input_shape)
    stem = StemBlock(input_tensor, [32, 16, 32], [(3, 3), (3, 3), (1, 1)], [(2, 2), (2, 2), (1, 1)], "relu",
                         weight_decay)
    regnetx_200MF = RegNetX(stem, [1, 1, 4, 7],
                            [[24, 24, 24], [56, 56, 56], [152, 152, 152], [368, 368, 368]],
                            [(3, 3), (3, 3), (3, 3), (3, 3)], [(2, 2), (2, 2), (2, 2), (2, 2)], [8, 8, 8, 8], "relu",
                            weight_decay)
    model = tf.keras.Model(inputs=[input_tensor], outputs=regnetx_200MF)
    model.summary()
    # get_flops(model)
    # model.save("RegNetX.h5")
    regnety_200MF = RegNetY(input_tensor, [1, 1, 4, 7],
                            [[24, 24, 24], [56, 56, 56], [152, 152, 152], [368, 368, 368]],
                            [(3, 3), (3, 3), (3, 3), (3, 3)], [(2, 2), (2, 2), (2, 2), (2, 2)], [8, 8, 8, 8], "relu",
                            weight_decay, [8, 8, 8, 8])
    model = tf.keras.Model(inputs=[input_tensor], outputs=regnety_200MF)
    model.summary()
    # get_flops(model)
    # model.save("RegNetY.h5")

    regnetz_200MF = RegNetZ(input_tensor, [1, 1, 4, 7],
                            [[24, 24, 24], [56, 56, 56], [152, 152, 152], [368, 368, 368]],
                            [(3, 3), (3, 3), (3, 3), (3, 3)], [(2, 2), (2, 2), (2, 2), (2, 2)], [8, 8, 8, 8], "relu",
                            weight_decay, [8, 8, 8, 8])
    model = tf.keras.Model(inputs=[input_tensor], outputs=regnetz_200MF)
    model.summary()
    # get_flops(model)
    # model.save("RegNetZ.h5")