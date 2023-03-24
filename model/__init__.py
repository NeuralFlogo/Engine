import model.architecture.classification
from model.architecture.convolutional import ConvolutionalArchitecture
from model.architecture.feed_forward import FeedForward
from model.models import SimpleModel
from model.RNN.RNN import RNN

architecture_conv = [{"kernel_conv": (5, 5), "in_channels": 10, "out_channels": 10, "activation": "ReLU",
                      "kernel_pool": (2, 2), "stride_pool": (2, 2), "padding_pool": (1, 1), "pooling_type": "Avg",
                      "stride_conv": (2, 2), "padding_conv": (1, 1)},
                     {"kernel_conv": (5, 1), "in_channels": 10, "out_channels": 10, "activation": "ReLU",
                      "kernel_pool": (2, 2), "stride_pool": (2, 2), "padding_pool": (1, 1), "pooling_type": "Avg",
                      "stride_conv": (2, 2), "padding_conv": (1, 1)},
                     {"kernel_conv": (5, 6), "in_channels": 10, "out_channels": 10, "activation": "ReLU",
                      "kernel_pool": (2, 2), "stride_pool": (2, 2), "padding_pool": (1, 1), "pooling_type": "Avg",
                      "stride_conv": (2, 2), "padding_conv": (1, 1)}]

pytorch_architecture = ConvolutionalArchitecture(architecture_conv).pytorch()
SimpleModel(pytorch_architecture)

architecture_feedforward = [{"input_dimension": 10, "output_dimension": 10, "activation": "ReLU"},
                            {"input_dimension": 10, "output_dimension": 10, "activation": "ReLU"}]
pytorch_architecture = FeedForward(architecture_feedforward).pytorch()
SimpleModel(pytorch_architecture)

architecture_recurrent = {"num_layers": 10, "hidden_size": 4, "input_size": 10, "block_type": "GRUCell",
                          "activation": "ReLU", "bias": True, "output_size": 2}
pytorch_architecture = RNN(architecture_recurrent).pytorch()

architecture_classification = {"name": "Softmax", "dimension": 10}
print(model.architecture.classification.Classification(architecture_classification).pytorch())
