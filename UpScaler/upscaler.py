import os
from abc import abstractmethod

import PIL
from PIL import Image

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
NEAREST = (Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST)

from urllib.parse import urlparse


def friendly_name(file: str):
    if "http" in file:
        file = urlparse(file).path

    file = os.path.basename(file)
    model_name, extension = os.path.splitext(file)
    return model_name


def walk_files(path, allowed_extensions=None):
    if not os.path.exists(path):
        return

    if allowed_extensions is not None:
        allowed_extensions = set(allowed_extensions)

    for root, _, files in os.walk(path, followlinks=True):
        for filename in files:
            if allowed_extensions is not None:
                _, ext = os.path.splitext(filename)
                if ext not in allowed_extensions:
                    continue

            yield os.path.join(root, filename)


def load_models(model_path: str, model_url: str = None, command_path: str = None, ext_filter=None, download_name=None,
                ext_blacklist=None) -> list:
    """
    A one-and done loader to try finding the desired models in specified directories.

    @param download_name: Specify to download from model_url immediately.
    @param model_url: If no other models are found, this will be downloaded on upscale.
    @param model_path: The location to store/find models in.
    @param command_path: A command-line argument to search for models in first.
    @param ext_filter: An optional list of filename extensions to filter by
    @return: A list of paths containing the desired model(s)
    """
    output = []

    try:
        places = []

        if command_path is not None and command_path != model_path:
            pretrained_path = os.path.join(command_path, 'experiments/pretrained_models')
            if os.path.exists(pretrained_path):
                print(f"Appending path: {pretrained_path}")
                places.append(pretrained_path)
            elif os.path.exists(command_path):
                places.append(command_path)

        places.append(model_path)

        for place in places:
            for full_path in walk_files(place, allowed_extensions=ext_filter):
                if os.path.islink(full_path) and not os.path.exists(full_path):
                    print(f"Skipping broken symlink: {full_path}")
                    continue
                if ext_blacklist is not None and any(full_path.endswith(x) for x in ext_blacklist):
                    continue
                if full_path not in output:
                    output.append(full_path)

        if model_url is not None and len(output) == 0:
            if download_name is not None:
                from basicsr.utils.download_util import load_file_from_url
                dl = load_file_from_url(model_url, places[0], True, download_name)
                output.append(dl)
            else:
                output.append(model_url)

    except Exception:
        pass

    return output


class Upscaler:
    name = None
    model_path = None
    model_name = None
    model_url = None
    enable = True
    filter = None
    model = None
    user_path = None
    tile = True

    scalers: []

    def __init__(self, models_path="model", device="cpu", no_half=True, ESRGAN_tile=192, ESRGAN_tile_overlap=8):
        self.mod_pad_h = None
        self.img = None
        self.output = None
        self.scale = 1
        self.tile_size = ESRGAN_tile
        self.tile_pad = ESRGAN_tile_overlap
        self.device = device
        self.half = not no_half

        self.pre_pad = 0
        self.mod_scale = None
        self.model_download_path = models_path

        if self.model_path is None and self.name:
            self.model_path = os.path.join(models_path, self.name)
        if self.model_path is not None and not os.path.exists(self.model_path):
            os.makedirs(self.model_path, exist_ok=True)

        # try:
        #     import cv2  # noqa: F401
        #     self.can_tile = True
        # except Exception:
        #     pass

    @abstractmethod
    def do_upscale(self, img: PIL.Image, selected_model: str):
        return img

    def upscale(self, img: PIL.Image, scale, selected_model: str = None):
        self.scale = scale
        dest_w = int(img.width * scale)
        dest_h = int(img.height * scale)

        for _ in range(3):
            shape = (img.width, img.height)
            img = self.do_upscale(img, selected_model)
            if shape == (img.width, img.height):
                break
            if img.width >= dest_w and img.height >= dest_h:
                break
        if img.width != dest_w or img.height != dest_h:
            img = img.resize((int(dest_w), int(dest_h)), resample=LANCZOS)
        return img

    @abstractmethod
    def load_model(self, path: str):
        pass

    def find_models(self, ext_filter=None) -> list:
        return load_models(model_path=self.model_path, model_url=self.model_url,
                           command_path=self.user_path)

    # def update_status(self, prompt):
    #     print(f"\nextras: {prompt}", file=shared.progress_print_out)


class UpscalerData:
    name = None
    data_path = None
    scale: int = 4
    scaler: Upscaler = None
    model: None

    def __init__(self, name: str, path: str, upscaler: Upscaler = None, scale: int = 4, model=None):
        self.name = name
        self.data_path = path
        self.local_data_path = path
        self.scaler = upscaler
        self.scale = scale
        self.model = model


class UpscalerLanczos(Upscaler):
    scalers = []

    def do_upscale(self, img, scale, selected_model=None):
        return img.resize((int(img.width * scale), int(img.height * scale)), resample=LANCZOS)

    def load_model(self, _):
        pass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Lanczos"
        self.scalers = [UpscalerData("Lanczos", None, self)]


class UpscalerNearest(Upscaler):
    def do_upscale(self, img, scale, selected_model=None):
        return img.resize((int(img.width * scale), int(img.height * scale)), resample=NEAREST)

    def load_model(self, _):
        pass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Nearest"
        self.scalers = [UpscalerData("Nearest", None, self)]
