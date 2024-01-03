from .Cv2Processor import InterpolationProcessor
from .RealESRGAN import RealESRGANProcessor, SwinIRProcessor
from .RealESRGAN import NearestProcessor, LanczosProcessor

__all__ = [
    InterpolationProcessor,
    RealESRGANProcessor,
    SwinIRProcessor,
    NearestProcessor,
    LanczosProcessor
]
