import compiled.model.layers as compiled_layers
from compiled.model.blocks.classification import CompiledClassificationBlock
from compiled.model.blocks.convolutional import CompiledConvolutionalBlock
from compiled.model.blocks.flatten import CompiledFlattenBlock
from compiled.model.blocks.linear import CompiledLinearBlock
from compiled.model.blocks.recurrent import CompiledRecurrentBlock
from compiled.model.blocks.residual import CompiledInputBlock, CompiledBodyBlock, CompiledOutputBlock
from pytorch.model.sections.link.classification import Classification
from pytorch.model.sections.link.flatten import Flatten
from pytorch.model.sections.processing.convolutional import ConvolutionalSection
from pytorch.model.sections.processing.feed_forward import FeedForwardSection
from pytorch.model.sections.processing.recurrent.recurrent import RecurrentSection
from pytorch.model.sections.processing.residual import ResidualSection

feed_forward = [
    CompiledLinearBlock(compiled_layers.linear.Linear(100, 10), compiled_layers.activation.Activation("ReLU")),
    CompiledLinearBlock(compiled_layers.linear.Linear(10, 2), compiled_layers.activation.Activation("ReLU"))]
FeedForwardSection(feed_forward).build()

flatten = CompiledFlattenBlock(compiled_layers.flatten.Flatten(10, 8))
Flatten(flatten).build()

classification = CompiledClassificationBlock(compiled_layers.classification.Classification("Softmax", 10))
Classification(classification).build()

convolutional = [CompiledConvolutionalBlock([compiled_layers.convolutional.Conv((), 10, 10, (), ()),
                                             compiled_layers.convolutional.Conv((), 10, 10, (), ()),
                                             compiled_layers.activation.Activation("ReLU"),
                                             compiled_layers.pool.Pool((), (), (), "Max")]),
                 CompiledConvolutionalBlock([compiled_layers.convolutional.Conv((), 10, 10, (), ()),
                                             compiled_layers.convolutional.Conv((), 10, 10, (), ()),
                                             compiled_layers.activation.Activation("ReLU"),
                                             compiled_layers.pool.Pool((), (), (), "Max")])]
ConvolutionalSection(convolutional).build()

residual = [CompiledInputBlock(compiled_layers.convolutional.Conv((), 10, 10, (), ()),
                               compiled_layers.pool.Pool((), (), (), "Max")),
            CompiledBodyBlock([compiled_layers.convolutional.Conv((), 10, 10, (), ()),
                                 compiled_layers.activation.Activation("ReLU"),
                                 compiled_layers.convolutional.Conv((), 10, 10, (), ())], 2),
            CompiledOutputBlock(compiled_layers.pool.Pool((), (), (), "Max"))]
ResidualSection(residual).build()

recurrent = [CompiledRecurrentBlock(3, 15, 5, "GRUCell", "ReLU", True)]
RecurrentSection(recurrent).build()
