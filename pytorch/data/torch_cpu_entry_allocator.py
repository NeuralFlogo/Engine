from framework.data.dataset.entry import Entry


class TorchCpuEntryAllocator:
    def allocate(self, entry: Entry):
        entry.inputs = entry.get_input().cpu()
        entry.outputs = entry.get_output().cpu()
