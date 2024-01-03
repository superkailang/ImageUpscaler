from PIL import Image
from srmd_ncnn_vulkan_python import Srmd


class UpscalerSrmd:
    def __call__(self, inputImg: Image, scale=2, size=None) -> Image:
        upscaler = Srmd(gpuid=0, scale=scale, noise=3)
        outImg = upscaler.process(inputImg)
        # if size is not None:
        #     outImg = outImg.resize(size, Image.Resampling.LANCZOS)
        return outImg
