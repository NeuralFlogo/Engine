from framework.structure.metadata import Metadata


class ModuleInterpreter:
    def generate(self, modules):
        structure, start_index, metadata = [], 0, Metadata()
        for module in modules:
            metadata.add(start_index, len(module))
            structure.extend(module.get_section())
            start_index += len(module)
        return structure, metadata
