"""Person 4 model adapters (lazy imports where heavy deps are optional)."""

from adapters.unet_keras import UnetKerasAdapter
from adapters.segformer_stub import SegFormerStubAdapter, SegFormerNotReadyError
from adapters.deeplabv3 import DeepLabV3Adapter

__all__ = [
    "UnetKerasAdapter",
    "UnetTorchAdapter",
    "SegFormerStubAdapter",
    "SegFormerNotReadyError",
    "SegFormerAdapter",
    "DeepLabV3Adapter",
]


def __getattr__(name: str):
    if name == "SegFormerAdapter":
        from adapters.segformer import SegFormerAdapter

        return SegFormerAdapter
    if name == "UnetTorchAdapter":
        from adapters.unet_torch import UnetTorchAdapter

        return UnetTorchAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
