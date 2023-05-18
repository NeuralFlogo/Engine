from framework.structure.sections.link.flatten import FlattenSection
from framework.structure.sections.processing.convolutional import ConvolutionalSection
from framework.structure.sections.processing.feed_forward import LinearSection
from framework.structure.sections.processing.recurrent import RecurrentSection
from framework.structure.sections.processing.residual import ResidualSection
from framework.structure.sections.link.classificationsection import ClassificationSection
import pytorch.structure.sections.processing.convolutional as convolutional
import pytorch.structure.sections.processing.feed_forward as feed_forward
import pytorch.structure.sections.processing.residual as residual
import pytorch.structure.sections.link.classification as classification
import pytorch.structure.sections.link.flatten as flatten
from pytorch.structure.sections.processing import recurrent


class PytorchGenerator:
    def generate(self, structure):
        result = []
        for section in structure:
            result.extend(self.__switch(section))
        return result

    def __switch(self, section):
        if type(section) == ConvolutionalSection: return convolutional.ConvolutionalSection(section).build()
        if type(section) == LinearSection: return feed_forward.FeedForwardSection(section).build()
        if type(section) == RecurrentSection: return recurrent.RecurrentSection(section).build()
        if type(section) == ResidualSection: return residual.ResidualSection(section).build()
        if type(section) == ClassificationSection: return classification.ClassificationSection(section).build()
        if type(section) == FlattenSection: return flatten.FlattenSection(section).build()
