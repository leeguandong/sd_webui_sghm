import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.hub import download_url_to_file
from .model.model import HumanMatting
from .model.utils import get_unknown_tensor_from_pred

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sghm = HumanMatting(backbone="resnet50")
sghm = nn.DataParallel(sghm).to(device).eval()


# 加载模型
def download_models(model_id):
    if "sghm" in model_id:
        url = "https://hf-mirror.com/endorno/SGHM/resolve/main/SGHM-ResNet50.pth"
    else:
        url = "https://huggingface.co/endorno/SGHM/resolve/main/SGHM-ResNet50.pth"
    models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir, exist_ok=True)
    sghm_checkpoint = os.path.join(models_dir, model_id)

    if not os.path.isfile(sghm_checkpoint):
        try:
            download_url_to_file(url, sghm_checkpoint)
        except Exception as e:
            print(f"{str(e)}")
    else:
        print("Model already exists")
    return sghm_checkpoint


sghm.load_state_dict(torch.load(download_models(model_id="SGHM")))

pil_to_tensor = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)

infer_size = 1280


def single_inference(model, img):
    if model == "SGHM":
        model = sghm

    h = img.height
    w = img.width
    if w >= h:
        rh = infer_size
        rw = int(w / h * infer_size)
    else:
        rw = infer_size
        rh = int(h / w * infer_size)
    rh = rh - rh % 64
    rw = rw - rw % 64

    img = pil_to_tensor(img)
    img = img[None, :, :, :].cuda()

    input_tensor = F.interpolate(img, size=(rh, rw), mode='bilinear')
    with torch.no_grad():
        pred = model(input_tensor)

    # progressive refine alpha
    alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
    pred_alpha = alpha_pred_os8.clone().detach()
    weight_os4 = get_unknown_tensor_from_pred(pred_alpha, rand_width=30, train_mode=False)
    pred_alpha[weight_os4 > 0] = alpha_pred_os4[weight_os4 > 0]
    weight_os1 = get_unknown_tensor_from_pred(pred_alpha, rand_width=15, train_mode=False)
    pred_alpha[weight_os1 > 0] = alpha_pred_os1[weight_os1 > 0]

    pred_alpha = pred_alpha.repeat(1, 3, 1, 1)
    pred_alpha = F.interpolate(pred_alpha, size=(h, w), mode='bilinear')
    alpha_np = pred_alpha[0].data.cpu().numpy().transpose(1, 2, 0)
    alpha_np = alpha_np[:, :, 0]

    # output segment
    pred_segment = pred['segment']
    pred_segment = F.interpolate(pred_segment, size=(h, w), mode='bilinear')
    segment_np = pred_segment[0].data.cpu().numpy().transpose(1, 2, 0)

    return alpha_np, segment_np
