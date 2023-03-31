class Block:
    Type = "block_type"
    HiddenSize = "hidden_size"


class Layers:
    Size = "layers_number"
    Bias = "layers_bias"


class Kernel:
    Convolutional = "kernel_conv"
    Pool = "kernel_pool"


class Stride:
    Convolutional = "stride_conv"
    Pool = "stride_pool"


class Padding:
    Convolutional = "padding_conv"
    Pool = "padding_pool"


class Pooling:
    type = "pooling_type"


class Channel:
    In = "channel_in"
    Out = "channel_out"


class Activation:
    name = "activation_name"
    dimension = "activation_dimension"


class Dimension:
    Start = "start_dim"
    End = "end_dim"
