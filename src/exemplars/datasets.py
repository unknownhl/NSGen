"""Dataset configs for computing exemplars.

The main function here is `dataset_hub`, which returns a mapping from dataset
name to a config specifying how to load it. The most important thing to know
is that the config takes a factory function for the dataset and arbitrary
kwargs to pass that factory. If a download URL is not specified, it expects
the dataset to live at $MILAN_DATA_DIR/dataset_name by default. See
`src/utils/hubs.py` for all the different options the configs support.
"""
import pathlib
from typing import Any, Mapping, Optional

from src import milannotations
from src.deps.netdissect import renormalize
from src.utils import hubs
from src.utils.typing import PathLike

import easydict
import torch
import torch.utils.data
import torchvision.datasets
import torchvision.transforms


KEYS = easydict.EasyDict(d=milannotations.KEYS)

class TensorDatasetOnDisk(torch.utils.data.TensorDataset):
    """Like `torch.utils.data.TensorDataset`, but tensors are pickled."""

    def __init__(self, root: PathLike, **kwargs: Any):
        """Load tensors from path and pass to `TensorDataset`.

        Args:
            root (PathLike): Root directory containing one or more .pth files
                of tensors.

        """
        loaded = []
        for child in pathlib.Path(root).iterdir():
            if not child.is_file() or not child.suffix == '.pth':
                continue
            tensors = torch.load(child, **kwargs)
            loaded.append(tensors)
        loaded = sorted(loaded,
                        key=lambda tensor: not tensor.dtype.is_floating_point)
        super().__init__(*loaded)


def default_dataset_configs(
        **others: hubs.DatasetConfig,  # Your overrides!
) -> Mapping[str, hubs.DatasetConfig]:
    """Return the default dataset configs."""
    configs = {
        'cifar10':
            hubs.DatasetConfig(
                torchvision.datasets.CIFAR10,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.Resize((32, 32)),
                    torchvision.transforms.CenterCrop(32),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                ])),
        # 'svhn':
        #     hubs.DatasetConfig(
        #         torchvision.datasets.SVHN,
        #         transform=torchvision.transforms.Compose([
        #             torchvision.transforms.Resize((32, 32)),
        #             torchvision.transforms.ToTensor(),
        #             torchvision.transforms.Normalize(
        #                 (0.4377, 0.4438, 0.4728), (0.1981, 0.2012, 0.1972))
        #         ])),
        'ImageNet':
            hubs.DatasetConfig(torchvision.datasets.ImageFolder,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Resize(256),
                                   torchvision.transforms.CenterCrop(224),
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                               ])),
    }
    configs.update(others)
    return configs


def default_dataset_hub(**others: hubs.DatasetConfig) -> hubs.DatasetHub:
    """Return configs for all datasets used in dissection."""
    configs = default_dataset_configs(**others)
    return hubs.DatasetHub(**configs)


def load(name: str,
         configs: Optional[Mapping[str, hubs.DatasetConfig]] = None,
         **kwargs: Any) -> torch.utils.data.Dataset:
    """Load the dataset.

    Args:
        name (str): The name of the dataset.
        configs (Optional[Mapping[str, hubs.DatasetConfig]], optional): Configs
            to load from. Defaults to those returned by default_dataset_hub().

    Returns:
        torch.utils.data.Dataset: The loaded dataset.

    """
    configs = configs or {}
    hub = default_dataset_hub(**configs)
    return hub.load(name, **kwargs)
