import io
import torch
import torch.nn as nn
import gradio as gr
import numpy as np

from enum import Enum
from PIL import Image, ImageOps
from PIL.Image import Image as PILImage
from modules.ui_components import FormRow
from modules import scripts_postprocessing
from sghm.inference import single_inference

models = [
    "None",
    "SGHM"
]


class ReturnType(Enum):
    BYTES = 0
    PILLOW = 1
    NDARRAY = 2


def fix_image_orientation(img: PILImage) -> PILImage:
    return ImageOps.exif_transpose(img)


class ScriptPostprocessingUpscale(scripts_postprocessing.ScriptPostprocessing):
    name = "sghm"
    order = 20000
    model = None

    def ui(self):
        with FormRow():
            model = gr.Dropdown(label="Remove human background", choices=models, value="None")
            return_mask = gr.Checkbox(label="Return mask", value=False)
            alpha_matting = gr.Checkbox(label="Alpha matting", value=False)

        return {
            "model": model,
            "return_mask": return_mask,
            "alpha_matting": alpha_matting
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, model, return_mask, alpha_matting):
        if not model or model == "None":
            return

        data = pp.image  # PIL
        if isinstance(data, PILImage):
            return_type = ReturnType.PILLOW
            img = data
        elif isinstance(data, bytes):
            return_type = ReturnType.BYTES
            img = Image.open(io.BytesIO(data))
        elif isinstance(data, np.ndarray):
            return_type = ReturnType.NDARRAY
            img = Image.fromarray(data)
        else:
            raise ValueError("Input type {} is not supported.".format(type(data)))
        # img = img.convert("RGB")

        # Fix image orientation
        img = fix_image_orientation(img)

        # 分割图返回的是mask，alpha图返回的扣完的图,在sghm中，会有分割图做辅助分支，但是此时的alpha其实等同于风格的unet中的mask
        alpha_np, mask_np = single_inference(model, img)  # (979,684) (979,684,1)
        alpha = Image.fromarray((alpha_np * 255).astype("uint8"), mode="L")
        mask = Image.fromarray(((np.squeeze(mask_np)) * 255).astype('uint8'), mode='L')
        alpha_matting_cutout = Image.new("RGBA", img.size)
        alpha_matting_cutout.paste(img, (0, 0), mask=alpha)

        if return_mask:
            if ReturnType.PILLOW == return_type:
                pp.image = alpha
            elif ReturnType.NDARRAY == return_type:
                pp.image = np.asarray(alpha)
            else:
                bio = io.BytesIO()
                alpha.save(bio, "PNG")
                bio.seek(0)
                pp.image = bio.read()
        else:
            if ReturnType.PILLOW == return_type:
                pp.image = alpha_matting_cutout
            elif ReturnType.NDARRAY == return_type:
                pp.image = np.asarray(alpha_matting_cutout)
            else:
                bio = io.BytesIO()
                alpha_matting_cutout.save(bio, "PNG")
                bio.seek(0)
                pp.image = bio.read()

        pp.info["SGHM"] = model
