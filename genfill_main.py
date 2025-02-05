import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import torch
from diffusers import AutoencoderKL, TCDScheduler 
from diffusers.models.model_loading_utils import load_state_dict
#from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download
from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline
from PIL import Image, ImageDraw
import traceback
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=SyntaxWarning)

from torch.cuda.amp import autocast  # Import mixed precision support

config_file = hf_hub_download(
    "xinsir/controlnet-union-sdxl-1.0",
    filename="config_promax.json",
)

config = ControlNetModel_Union.load_config(config_file)
controlnet_model = ControlNetModel_Union.from_config(config)
model_file = hf_hub_download(
    "xinsir/controlnet-union-sdxl-1.0",
    filename="diffusion_pytorch_model_promax.safetensors",
)
state_dict = load_state_dict(model_file)
model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
    controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
)
model.to(device="cuda", dtype=torch.float16)

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
).to("cuda")

pipe = StableDiffusionXLFillPipeline.from_pretrained(
    "SG161222/RealVisXL_V5.0_Lightning",
    torch_dtype=torch.float16,
    vae=vae,
    controlnet=model,
    variant="fp16",
).to("cuda")

pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)


def fill_image(prompt, image, paste_back=True):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    final_prompt = f"{prompt} , high quality, 4k, avoid bad anatomy, avoid bad proportions, avoid disfigured, avoid deformed, avoid blurry, avoid cropped, avoid duplicate, avoid error, avoid extra limbs, avoid malformed, avoid mutated, avoid mutilated, avoid nudity, avoid out of frame, avoid low quality, avoid lowres, avoid long neck, avoid jpeg artifacts, avoid gross proportions, avoid worst quality, avoid unflattering"
    
    with torch.inference_mode():
        with autocast():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(final_prompt, "cuda", True)

    if "background" not in image or "layers" not in image or not image["layers"]:
        raise ValueError("Background image or mask is missing")

    source = image["background"]
    mask = image["layers"][0]

    # Pastikan ukuran mask sama dengan background
    mask = mask.resize(source.size, Image.LANCZOS)

    # Pastikan semua gambar dalam mode RGBA
    if source.mode != "RGBA":
        source = source.convert("RGBA")

    if mask.mode != "RGBA":
        mask = mask.convert("RGBA")

    # Ambil channel alpha sebagai mask biner
    alpha_channel = mask.split()[3]
    binary_mask = alpha_channel.point(lambda p: p > 0 and 255)

    # Pastikan ukuran binary_mask cocok dengan cnet_image
    cnet_image = source.copy()
    if binary_mask.size != cnet_image.size:
        print(f"Resizing binary_mask from {binary_mask.size} to {cnet_image.size}")
        binary_mask = binary_mask.resize(cnet_image.size, Image.LANCZOS)

    try:
        cnet_image.paste(0, (0, 0), binary_mask)
    except Exception as e:
        print(f"Error saat paste mask: {e}")
        return

    # Proses dengan model
    for image in pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image,
    ):
        yield image, cnet_image

    if paste_back:
        image = image.convert("RGBA")
        cnet_image.paste(image, (0, 0), binary_mask)
    else:
        cnet_image = image

    yield source, cnet_image
