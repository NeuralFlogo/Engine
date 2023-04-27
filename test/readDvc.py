from pytorch.preprocesing.dvc_utils import *

PATH = "C:/Users/Joel/.aws/credentials"

remote_url = "https://github.com/Joeel71/test-dvc"
remote_path = "data/images"
set_keys(PATH)
parameters = {
    "batch_size": 10,
    "proportion": 0.8,
    "preprocessing": ["one-hot"] * 22,
    "size": 50,
    "mean": 0,
    "std": 1
}
train_loader, test_loader = read_from_dvc(remote_path, remote_url, "images", parameters)


