from UpScaler import UpscalerRealESRGAN, UpscalerSwinIR, UpscalerLanczos, UpscalerNearest
from PIL import Image


def scale(im_rgb: Image.Image, size):
    upscale_by = max(size[0] / im_rgb.width, size[1] / im_rgb.height)
    return upscale_by


class RealESRGANProcessor:
    def __init__(self, models_path="model", select_model="R-ESRGAN 4x+"):
        self.scaler = UpscalerRealESRGAN(models_path=models_path)
        self.select_model = select_model

    def __call__(self, im_rgb: Image.Image, size=(1024, 1024)):
        return self.scaler.upscale(im_rgb, scale(im_rgb, size), self.select_model)


class SwinIRProcessor:
    def __init__(self, select_model="SwinIR 4x", models_path="model"):
        self.scaler = UpscalerSwinIR(models_path=models_path)
        self.select_model = select_model

    def __call__(self, im_rgb: Image.Image, size=(1024, 1024)):
        return self.scaler.upscale(im_rgb, scale(im_rgb, size), self.select_model)


class NearestProcessor:
    def __init__(self):
        self.scaler = UpscalerNearest()

    def __call__(self, im_rgb: Image.Image, size=(1024, 1024)):
        return self.scaler.do_upscale(im_rgb, scale(im_rgb, size))


class LanczosProcessor:
    def __init__(self):
        self.scaler = UpscalerLanczos()

    def __call__(self, im_rgb: Image.Image, size=(1024, 1024)):
        return self.scaler.do_upscale(im_rgb, scale(im_rgb, size))