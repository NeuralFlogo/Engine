from framework.data.dataset.entry import Entry


class TorchGpuEntryAllocator:
    def allocate(self, entry: Entry):
        entry.inputs = entry.get_input().cuda()
        entry.outputs = entry.get_output().cuda()

    def deallocate(self, entry: Entry):
        entry.inputs = entry.get_input().cpu()
        entry.outputs = entry.get_output().cpu()
