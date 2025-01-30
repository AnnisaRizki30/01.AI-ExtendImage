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


def prepare_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom, target_ratio=None):
    target_size = (width, height)

    try:
        # Calculate the scaling factor to fit the image within the target size
        scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)

        # Resize the source image to fit within target size
        source = image.resize((new_width, new_height), Image.LANCZOS)

        # Apply resize option using percentages
        if resize_option == "Full":
            resize_percentage = 100
        elif resize_option == "50%":
            resize_percentage = 50
        elif resize_option == "33%":
            resize_percentage = 33
        elif resize_option == "25%":
            resize_percentage = 25
        else:  # Custom
            resize_percentage = custom_resize_percentage

        # Calculate new dimensions based on percentage
        resize_factor = resize_percentage / 100
        new_width = int(source.width * resize_factor)
        new_height = int(source.height * resize_factor)

        # Ensure minimum size of 64 pixels
        new_width = max(new_width, 64)
        new_height = max(new_height, 64)

        # Resize the image
        source = source.resize((new_width, new_height), Image.LANCZOS)

        # Calculate the overlap in pixels based on the percentage
        overlap_x = int(new_width * (overlap_percentage / 100))
        overlap_y = int(new_height * (overlap_percentage / 100))

        # Ensure minimum overlap of 1 pixel
        overlap_x = max(overlap_x, 1)
        overlap_y = max(overlap_y, 1)

        # Calculate margins based on alignment
        if alignment == "Middle":
            margin_x = (target_size[0] - new_width) // 2
            margin_y = (target_size[1] - new_height) // 2
        elif alignment == "Left":
            margin_x = 0
            margin_y = (target_size[1] - new_height) // 2
        elif alignment == "Right":
            margin_x = target_size[0] - new_width
            margin_y = (target_size[1] - new_height) // 2
        elif alignment == "Top":
            margin_x = (target_size[0] - new_width) // 2
            margin_y = 0
        elif alignment == "Bottom":
            margin_x = (target_size[0] - new_width) // 2
            margin_y = target_size[1] - new_height

        # Adjust margins to eliminate gaps
        margin_x = max(0, min(margin_x, target_size[0] - new_width))
        margin_y = max(0, min(margin_y, target_size[1] - new_height))

        # Create a new background image and paste the resized source image
        background = Image.new('RGB', target_size, (255, 255, 255))
        background.paste(source, (margin_x, margin_y))

        # Create the mask
        mask = Image.new('L', target_size, 255)
        mask_draw = ImageDraw.Draw(mask)

        # Calculate overlap areas
        white_gaps_patch = 2

        left_overlap = margin_x + overlap_x if overlap_left else margin_x + white_gaps_patch
        right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width - white_gaps_patch
        top_overlap = margin_y + overlap_y if overlap_top else margin_y + white_gaps_patch
        bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height - white_gaps_patch
        
        if alignment == "Left":
            left_overlap = margin_x + overlap_x if overlap_left else margin_x
        elif alignment == "Right":
            right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width
        elif alignment == "Top":
            top_overlap = margin_y + overlap_y if overlap_top else margin_y
        elif alignment == "Bottom":
            bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height


        # Draw the mask
        mask_draw.rectangle([
            (left_overlap, top_overlap),
            (right_overlap, bottom_overlap)
        ], fill=0)

        return background, mask

    except Exception as e:
        # Capture the traceback and raise a detailed error
        print(f"Error encountered while processing the image: {e}")
        traceback.print_exc()
        raise e


def inference_extend_image(image, num_inference_steps=8, target_ratio=None, custom_width=None, custom_height=None, prompt_input=None):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    alignment = "Bottom"

    if target_ratio == "Square":
        width = 1024
        height = 1024
    elif target_ratio == "Landscape":
        width = 1280
        height = 720
    elif target_ratio == "Potrait":
        width = 720
        height = 1024
    elif target_ratio == "Instagram Post":
        width = 1080
        height = 1080
    elif target_ratio == "Instagram Stories":
        width = 800
        height = 1280
    elif target_ratio == "Custom":
        width = custom_width
        height = custom_height

    resize_option = "Full"
    overlap_percentage = 10
    custom_resize_percentage = 50
    overlap_left = False 
    overlap_right = False
    overlap_top = False
    overlap_bottom = False
        
    background, mask = prepare_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom, target_ratio)

    cnet_image = background.copy()
    cnet_image.paste(0, (0, 0), mask)

    final_prompt = "high quality, no nudity, no extra limbs, no disfigured bodies"
    if prompt_input and prompt_input.strip() != "":
        final_prompt += ", " + prompt_input.strip()

    # Encoding prompt menggunakan mixed precision
    with torch.inference_mode():
        with autocast():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(final_prompt, "cuda", True)


    for image in pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image,
        num_inference_steps=num_inference_steps
    ):
        yield cnet_image, image

    image = image.convert("RGBA")
    cnet_image.paste(image, (0, 0), mask)

    yield background, cnet_image
