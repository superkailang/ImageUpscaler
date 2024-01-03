from PIL import Image
from waifu2x_ncnn_vulkan_python import Waifu2x


class UpscalerWaifu2x:
    def __call__(self, inputImg: Image, scale=2, size=None) -> Image:
        waifu2x = Waifu2x(gpuid=0, scale=scale, noise=3)
        outImg = waifu2x.process(inputImg)
        # if size is not None:
        #     outImg = outImg.resize(size, Image.Resampling.LANCZOS)
        return outImg


# if __name__ == '__main__':
#     up = UpscalerWaifu2x()
#     up(Image.open("./data/bg.png"))
