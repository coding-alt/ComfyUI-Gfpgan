import os
import cv2
import glob
import torch
import numpy as np
from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import folder_paths

# 获取 ComfyUI 路径
comfy_path = os.path.dirname(folder_paths.__file__)
model_root = f'{comfy_path}/models'

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Converts a tensor to a ndarray with proper scaling and type conversion."""
    return np.clip(255.0 * tensor.detach().cpu().numpy(), 0, 255).astype(np.uint8)

def tensor2np(tensor: torch.Tensor) -> list[np.ndarray]:
    """Converts a batch of tensors to a list of NumPy arrays."""
    def single_tensor2np(t: torch.Tensor) -> np.ndarray:
        t = t.squeeze()  # Remove any singleton dimensions
        if t.ndim == 2:  # (H, W) for masks
            return to_numpy(t)
        elif t.ndim == 3:  # (C, H, W) for RGB/RGBA
            if t.shape[0] in [1, 3, 4]:  # Channel-first format
                t = t.permute(1, 2, 0)
            return to_numpy(t)
        else:
            raise ValueError(f"Invalid tensor shape: {t.shape}")
    return [single_tensor2np(tensor[i]) for i in range(tensor.shape[0])]

def initialize_upsampler(bg_upsampler_model: str, upscale: int, bg_tile: int):
    """Initialize the background upsampler using RealESRGAN."""
    if bg_upsampler_model != "None":
        bg_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        return RealESRGANer(
            scale=upscale,
            model_path=f"{model_root}/upscale_models/{bg_upsampler_model}",
            model=bg_model,
            tile=bg_tile,
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available()
        )
    return None

def initialize_restorer(model_path: str, upscale: int, arch: str, channel_multiplier: int, bg_upsampler):
    """Initialize the GFPGAN restorer."""
    return GFPGANer(
        model_path=f"{model_root}/facerestore_models/{model_path}",
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler
    )

class GFPGANNode:
    @classmethod
    def INPUT_TYPES(cls):
        # 动态加载 GFPGAN 模型文件
        gfpgan_models = glob.glob(f"{model_root}/facerestore_models/GFPGAN*.pth")
        gfpgan_model_choices = [os.path.basename(path) for path in gfpgan_models]
        if not gfpgan_model_choices:
            gfpgan_model_choices = ["None"]

        # 动态加载 RealESRGAN 模型文件
        realesrgan_model_path = f"{model_root}/upscale_models/RealESRGAN_x2plus.pth"
        realesrgan_choice = [os.path.basename(realesrgan_model_path)] if os.path.exists(realesrgan_model_path) else ["None"]

        return {
            "required": {
                "image": ("IMAGE",),
                "model_path": (gfpgan_model_choices, {"default": gfpgan_model_choices[0]}),
                "upscale": ("INT", {"default": 2, "min": 1, "max": 8}),
                "arch": ("STRING", {"default": "clean", "choices": ["original", "clean", "RestoreFormer"]}),
                "channel_multiplier": ("INT", {"default": 2, "choices": [1, 2]}),
                "bg_tile": ("INT", {"default": 400, "min": 0, "max": 2000}),
                "bg_upsampler_model": (realesrgan_choice, {"default": realesrgan_choice[0]}),
                "weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "GFPGAN"

    def process_image(self, image, model_path, upscale, arch, channel_multiplier, bg_tile, bg_upsampler_model, weight):
        # Convert input tensor to NumPy image
        pimage = tensor2np(image)[0]
        source_img = cv2.cvtColor(np.array(pimage), cv2.COLOR_RGB2BGR)

        # Initialize background upsampler
        bg_upsampler = initialize_upsampler(bg_upsampler_model, upscale, bg_tile)

        # Initialize GFPGAN restorer
        restorer = initialize_restorer(model_path, upscale, arch, channel_multiplier, bg_upsampler)

        # Restore image
        try:
            _, _, restored_img = restorer.enhance(
                source_img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=weight,
            )
        except Exception as e:
            raise RuntimeError(f"GFPGAN failed to process the image: {e}")

        if restored_img is None:
            raise RuntimeError("GFPGAN failed to generate a restored image.")
        
        # Convert restored image to tensor format
        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        restored_img = torch.from_numpy(np.array(restored_img).astype(np.float32) / 255.0)[None, ]

        return (restored_img,)


NODE_CLASS_MAPPINGS = {
    "GFPGANNode": GFPGANNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GFPGANNode": "GFPGAN Processor"
}