import model.model.layers as compiled_layers
from model.model.blocks.classification import FlogoClassificationBlock
from model.model.blocks.convolutional import FlogoConvolutionalBlock
from model.model.blocks.flatten import FlogoFlattenBlock
from model.model.blocks.linear import FlogoLinearBlock
from model.model.blocks.recurrent import FlogoRecurrentBlock
from model.model.blocks.residual import FlogoInputBlock, CompiledBodyBlock, CompiledOutputBlock
from pytorch.model.sections.link.classification import Classification
from pytorch.model.sections.link.flatten import Flatten
from pytorch.model.sections.processing.convolutional import ConvolutionalSection
from pytorch.model.sections.processing.feed_forward import FeedForwardSection
from pytorch.model.sections.processing.recurrent.recurrent import RecurrentSection
from pytorch.model.sections.processing.residual import ResidualSection

feed_forward = [
    FlogoLinearBlock([compiled_layers.linear.Linear(100, 10), compiled_layers.activation.Activation("ReLU")]),
    FlogoLinearBlock([compiled_layers.linear.Linear(10, 2), compiled_layers.activation.Activation("ReLU")])]
FeedForwardSection(feed_forward).build()

flatten = FlogoFlattenBlock(compiled_layers.flatten.Flatten(10, 8))
Flatten(flatten).build()

classification = FlogoClassificationBlock(compiled_layers.classification.Classification("Softmax", 10))
Classification(classification).build()

convolutional = [FlogoConvolutionalBlock([compiled_layers.convolutional.Conv((), 10, 10, (), ()),
                                          compiled_layers.convolutional.Conv((), 10, 10, (), ()),
                                          compiled_layers.activation.Activation("ReLU"),
                                          compiled_layers.pool.Pool((), (), (), "Max")]),
                 FlogoConvolutionalBlock([compiled_layers.convolutional.Conv((), 10, 10, (), ()),
                                          compiled_layers.convolutional.Conv((), 10, 10, (), ()),
                                          compiled_layers.activation.Activation("ReLU"),
                                          compiled_layers.pool.Pool((), (), (), "Max")])]
ConvolutionalSection(convolutional).build()

residual = [FlogoInputBlock(compiled_layers.convolutional.Conv((), 10, 10, (), ()),
                            compiled_layers.pool.Pool((), (), (), "Max")),
            CompiledBodyBlock([compiled_layers.convolutional.Conv((), 10, 10, (), ()),
                               compiled_layers.activation.Activation("ReLU"),
                               compiled_layers.convolutional.Conv((), 10, 10, (), ())], 2),
            CompiledOutputBlock(compiled_layers.pool.Pool((), (), (), "Max"))]
ResidualSection(residual).build()

recurrent = [FlogoRecurrentBlock(3, 15, 5, "GRUCell", "ReLU", True)]
RecurrentSection(recurrent).build()
