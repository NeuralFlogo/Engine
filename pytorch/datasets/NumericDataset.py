import torch
from torch.utils.data import Dataset

from pytorch.preprocesing.NumericProcessor import *

transformer_functions = {
    "one-hot": one_hot_encode,
    "standardize": standardize,
    "normalize": normalize,
    "to-number": to_number
}


def transform_x(data, parameters):
    transform_data = []
    for col in range(data.shape[1]):
        transform_data.append(transformer_functions.get(parameters[col])((data[:, col])))
    return to_tensor(transform_data)


def transform_y(data, parameters):
    return torch.tensor(transformer_functions.get(parameters[-1])(data), dtype=torch.float32)


def to_tensor(array):
    if len(array[0].shape) == 2:
        return torch.tensor(np.concatenate(array, axis=1), dtype=torch.float32)
    else:
        array = np.vstack(array)
        return torch.tensor(np.transpose(array), dtype=torch.float32)


class NumericDataset(Dataset):
    def __init__(self, x, y, parameters):
        self.x = transform_x(x, parameters)
        self.y = transform_y(y, parameters)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y
