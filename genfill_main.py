import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, TCDScheduler 
from diffusers.models.model_loading_utils import load_state_dict
# from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download
# from diffusers import StableDiffusionInpaintPipeline
from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline
from PIL import Image, ImageDraw
import traceback
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=SyntaxWarning)
from torch.cuda.amp import autocast  

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


def fill_image(prompt, image):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    final_prompt = f"{prompt}, high quality, 4k, avoid bad anatomy, avoid bad proportions, avoid disfigured, avoid deformed, avoid blurry, avoid cropped, avoid duplicate, avoid error, avoid extra limbs, avoid malformed, avoid mutated, avoid mutilated, avoid nudity, avoid out of frame, avoid low quality, avoid lowres, avoid long neck, avoid jpeg artifacts, avoid gross proportions, avoid worst quality, avoid unflattering"

    # Encode prompt embeddings untuk ControlNet
    with torch.inference_mode():
        with torch.autocast("cuda"):
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(final_prompt, "cuda", True)

    # Gambar asli dan mask
    source = image["background"]  # Gambar asli
    mask = image["layers"][0]  # Masking area

    # Pastikan mask memiliki kanal alpha
    alpha_channel = mask.split()[3]  # Kanal alpha
    binary_mask = alpha_channel.point(lambda p: 255 if p > 0 else 0)

    # Salin gambar asli untuk manipulasi
    cnet_image = source.copy()

    # Hapus area mask dari gambar input (diisi transparan)
    cnet_image.paste(0, (0, 0), binary_mask)

    # Proses gambar dengan model
    generated_images = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image,
    ).images

    # Ambil gambar hasil generasi
    generated_image = generated_images[0].convert("RGBA")

    # Gabungkan hasil generasi dengan gambar asli
    final_image = source.convert("RGBA")
    final_image.paste(generated_image, (0, 0), binary_mask)

    return final_image

# class Inpainting_Model(nn.Module):
#     def __init__(self,model_cards = "stabilityai/stable-diffusion-2-inpainting",**kwargs):
#         super().__init__()
#         is_cuda = torch.cuda.is_available()
#         self.device = "cuda" if  is_cuda else "cpu"
#         self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
#                                 model_cards,
#                                 torch_dtype=torch.float16 if is_cuda else torch.float32,
#                                 ).to(self.device)


#     def forward(self, prompt, image, mask,
#                 steps=50,
#                 guidance_scale=7.5,
#                 neg_prompt="bad anatomy, bad proportions, disfigured, deformed, blurry, cropped, duplicate, error, extra limbs, malformed, mutated, mutilated, nudity, out of frame, low quality, lowres, long neck, jpeg artifacts, gross proportions, worst quality, unflattering",
#                 num_samples=1, seed=0):
#         generator = torch.Generator(device=self.device).manual_seed(seed)

#         with torch.inference_mode():  
#             with torch.autocast(device_type=self.device, dtype=torch.float16): 
#                 output_img = self.pipeline(
#                     prompt=prompt,
#                     image=image,
#                     mask_image=mask,
#                     width=image.width,
#                     height=image.height,
#                     num_inference_steps=steps,
#                     guidance_scale=guidance_scale,
#                     negative_prompt=neg_prompt,
#                     generator=generator,
#                     num_images_per_prompt=num_samples,
#                 ).images
#         return output_img
