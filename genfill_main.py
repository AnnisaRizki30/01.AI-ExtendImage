import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import torch
import torch.nn as nn
from torch.cuda.amp import autocast  
# from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
import traceback
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=SyntaxWarning)

class Inpainting_Model(nn.Module):
    def __init__(self,model_cards = "stabilityai/stable-diffusion-2-inpainting",**kwargs):
        super().__init__()
        is_cuda = torch.cuda.is_available()
        self.device = "cuda" if  is_cuda else "cpu"
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                                model_cards,
                                torch_dtype=torch.float16 if is_cuda else torch.float32,
                                ).to(self.device)


    def forward(self, prompt, image, mask,
                negative_prompt,
                steps=50,
                guidance_scale=7.5,
                num_samples=1, seed=0):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.inference_mode():  
            with torch.autocast(device_type=self.device, dtype=torch.float16): 
                output_img = self.pipeline(
                    prompt=prompt,
                    image=image,
                    mask_image=mask,
                    width=image.width,
                    height=image.height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,  
                    generator=generator,
                    num_images_per_prompt=num_samples,
                ).images
        return output_img
