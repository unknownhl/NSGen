"""Configs describing how to compute exemplars for each model.

The most important function here is `model_hub`, which returns a mapping
from model name (formatted as <model architecture>/<training dataset>)
to a special config object. The config object is described more in
`src/utils/hubs.py`, but the most important thing to know is it takes
an arbitrary factory function for the model and, optionally, will look
for pretrained weights at $MILAN_MODELS_DIR/model_name.pth if
load_weights=True (though this path can be overwritten at runtime).

Additionally, the configs allow you to specify the *layers* to compute
exemplars for (by default, all of them). These must be a fully specified
path to the torch submodule, as they will be read using PyTorch hooks. To
see the full list of possible layers for your model, look at
`your_model.named_parameters()`.
"""
import dataclasses
from typing import Any, Mapping, Optional, Sequence, Tuple

from src import milannotations
from src.deps.netdissect import renormalize
from src.exemplars import transforms
from src.utils import hubs
from src.utils.typing import Layer

import easydict
from torch import nn

# We don't host most of these models, either the NetDissect team does
# or the torchvision people.
HOST = 'https://dissect.csail.mit.edu/models'

KEYS = easydict.EasyDict(d=milannotations.KEYS)

LAYERS = easydict.EasyDict()
LAYERS.MOBILENET_V2 = (f'features.{index}' for index in range(0, 19, 2))
# LAYERS.RESNET18 = ('conv1', 'layer1', 'layer2', 'layer3', 'layer4')
LAYERS.RESNET50 = ('conv1', 'layer1', 'layer2', 'layer3', 'layer4')
# LAYERS.RESNET50 = tuple(f'layer4.2.conv{index}' for index in range(1,4))
LAYERS.VGG16_BN = tuple(f'features.{index}' for index in (34, 37, 40))


@dataclasses.dataclass(frozen=True)
class ModelExemplarsConfig:
    """Generic dissection configuration."""

    k: Optional[int] = None
    quantile: Optional[float] = None
    output_size: Optional[int] = None
    batch_size: Optional[int] = None
    image_size: Optional[int] = None
    renormalizer: Optional[renormalize.Renormalizer] = None

    @property
    def kwargs(self) -> Mapping[str, Any]:
        """Convert the config to kwargs."""
        kwargs = {}
        for key, value in vars(self).items():
            if value is not None:
                kwargs[key] = value
        return kwargs


TransformInputs = transforms.TransformToTuple
TransformHiddens = transforms.TransformToTensor
TransformOutputs = transforms.TransformToTensor


@dataclasses.dataclass(frozen=True)
class DiscriminativeModelExemplarsConfig(ModelExemplarsConfig):
    """Dissection configuration for a discriminative model."""

    transform_inputs: Optional[TransformInputs] = None
    transform_hiddens: Optional[TransformHiddens] = None


@dataclasses.dataclass(frozen=True)
class GenerativeModelExemplarsConfig(ModelExemplarsConfig):
    """Dissection configuration for a model."""

    transform_inputs: Optional[TransformInputs] = None
    transform_hiddens: Optional[TransformHiddens] = None
    transform_outputs: Optional[TransformOutputs] = None

    # Special property: generative models want a dataset of representations,
    # not the dataset of images they were trained on. This is required.
    dataset: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the config."""
        if self.dataset is None:
            raise ValueError('GenerativeModelExemplarsConfig requires '
                             'dataset to be set')

    @property
    def kwargs(self) -> Mapping[str, Any]:
        """Convert the config to kwargs."""
        kwargs = dict(super().kwargs)
        kwargs.pop('dataset', None)
        return kwargs


@dataclasses.dataclass
class ModelConfig(hubs.ModelConfig):
    """A model config that also stores dissection configuration."""

    def __init__(self,
                 *args: Any,
                 layers: Optional[Sequence[Layer]] = None,
                 exemplars: Optional[ModelExemplarsConfig] = None,
                 **kwargs: Any):
        """Initialize the config.

        Args:
            exemplars (Optional[Mapping[str, Any]]): Exemplars options.

        """
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.exemplars = exemplars or ModelExemplarsConfig()


def default_model_configs(**others: ModelConfig) -> Mapping[str, ModelConfig]:
    """Return the default model configs."""

    configs = {

    }
    configs.update(others)

    return configs


def default_model_hub(**others: ModelConfig) -> hubs.ModelHub:
    """Return configs for all models for which we can extract exemplars."""
    configs = default_model_configs(**others)
    return hubs.ModelHub(**configs)


Model = Tuple[nn.Module, Sequence[Layer], ModelConfig]


def load(name: str,
         configs: Optional[Mapping[str, ModelConfig]] = None,
         **kwargs: Any) -> Model:
    """Load the model and also return its layers and config.

    Args:
        name (str): Model config name.
        configs (Optional[Mapping[str, ModelConfig]], optional): Model configs
            to use when loading models, in addition to those returned by
            default_model_hub(). Defaults to just those returned by
            default_model_hub().

    Returns:
        Model: The loaded model, it's default exemplar-able layers, and its
            config.

    """
    configs = configs or {}
    hub = default_model_hub(**configs)
    model = hub.load(name, **kwargs)

    config = hub.configs[name]
    assert isinstance(config, ModelConfig), 'unknown config type'
    layers = config.layers
    if layers is None:
        layers = [key for key, _ in model.named_children()]

    return model, layers, config
