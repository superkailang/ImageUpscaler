from pyanime4k import ac
from PIL import Image
import numpy as np


# import pyanime4k
# import pathlib
# pyanime4k.upscale_images(pathlib.Path('/home/notebook/diffusers/image/bg.png'),GPU_mode=True)


class UpscalerPyanime4k:
    def __call__(self, inputImg: Image, scale=2, size=None) -> Image:
        parameters = ac.Parameters()
        # enable HDN for ACNet
        parameters.HDN = True

        a = ac.AC(
            # managerList=ac.ManagerList([ac.OpenCLACNetManager(pID=0, dID=0)]),
            type=ac.ProcessorType.GPU
        )

        # load image from file
        # a.load_image(r"/home/notebook/diffusers/image/bg.png")

        # start processing
        # a.process()

        # im = Image.open("/home/notebook/diffusers/image/bg.png")

        arr = a.proccess_image_with_numpy(np.array(inputImg))

        status = a.get_process_status()
        print(status)
        im = Image.fromarray(arr)
        if size is not None:
            im = im.resize(size, Image.Resampling.LANCZOS)
        return im
