from .upscaler import UpscalerNearest, UpscalerLanczos
from .realesrgan_model import UpscalerRealESRGAN
from .swinir_model import UpscalerSwinIR

__all__ = [
    UpscalerRealESRGAN, UpscalerLanczos, UpscalerNearest, UpscalerSwinIR
]
