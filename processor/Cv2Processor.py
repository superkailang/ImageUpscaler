from PIL import Image


class InterpolationProcessor:
    def __call__(self, im_rgb: Image.Image, size=(1024, 1024)):
        result = im_rgb.resize(size)
        return result
