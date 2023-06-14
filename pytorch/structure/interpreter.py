from framework.structure.metadata import Metadata
from framework.structure.sections.link.flatten import FlattenSection
from framework.structure.sections.processing.convolutional import ConvolutionalSection
from framework.structure.sections.processing.linear import LinearSection
from framework.structure.sections.processing.recurrent import RecurrentSection
from framework.structure.sections.processing.residual import ResidualSection
from framework.structure.sections.link.classificationsection import ClassificationSection
import pytorch.structure.sections.processing.convolutional as convolutional
import pytorch.structure.sections.processing.linear as linear_section
import pytorch.structure.sections.processing.residual as residual
import pytorch.structure.sections.link.classification as classification
import pytorch.structure.sections.link.flatten as flatten
from pytorch.structure.sections.processing import recurrent


class PytorchInterpreter:
    def generate(self, definition):
        torch_structure, metadata, start_index = [], Metadata(), 0
        for section in definition:
            section = self.__switch(section)
            metadata.add(start_index, len(section))
            start_index += len(section)
            torch_structure.extend(section)
        return torch_structure, metadata

    def __switch(self, section):
        if type(section) == ConvolutionalSection: return convolutional.ConvolutionalSection(section).build()
        if type(section) == LinearSection: return linear_section.LinearSection(section).build()
        if type(section) == RecurrentSection: return recurrent.RecurrentSection(section).build()
        if type(section) == ResidualSection: return residual.ResidualSection(section).build()
        if type(section) == ClassificationSection: return classification.ClassificationSection(section).build()
        if type(section) == FlattenSection: return flatten.FlattenSection(section).build()
