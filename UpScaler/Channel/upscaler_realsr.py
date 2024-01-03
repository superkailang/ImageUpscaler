from PIL import Image
from realsr_ncnn_vulkan_python import RealSR


class UpscalerRealsr:
    def __call__(self, inputImg: Image, scale=2, size=None) -> Image:
        upscaler = RealSR(0, scale=scale, noise=3)
        outImg = upscaler.process(inputImg)
        # if size is not None:
        #     outImg = outImg.resize(size, Image.Resampling.LANCZOS)
        return outImg
