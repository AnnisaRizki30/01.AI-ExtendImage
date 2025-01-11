import gradio as gr
import torch
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download

from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

from PIL import Image, ImageDraw

# Load models
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

def infer(image, num_inference_steps=8, prompt_input=None):
    target_size = (1280, 720)

    source = image

    if source.width < target_size[0] and source.height < target_size[1]:
        scale_factor = min(target_size[0] / source.width, target_size[1] / source.height)
        new_width = int(source.width * scale_factor)
        new_height = int(source.height * scale_factor)
        source = source.resize((new_width, new_height), Image.LANCZOS)

    if source.width > target_size[0] or source.height > target_size[1]:
        scale_factor = min(target_size[0] / source.width, target_size[1] / source.height)
        new_width = int(source.width * scale_factor)
        new_height = int(source.height * scale_factor)
        source = source.resize((new_width, new_height), Image.LANCZOS)

    margin_x = (target_size[0] - source.width) // 2
    margin_y = (target_size[1] - source.height) // 2

    background = Image.new('RGB', target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))

    mask = Image.new('L', target_size, 255)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle([
        (margin_x, margin_y),
        (margin_x + source.width, margin_y + source.height)
    ], fill=0)

    cnet_image = background.copy()
    cnet_image.paste(0, (0, 0), mask)

    final_prompt = "high quality"
    if prompt_input and prompt_input.strip():
        final_prompt += ", " + prompt_input

    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(final_prompt, "cuda", True)

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

def clear_result():
    return gr.update(value=None)

css = """
.gradio-container {
    width: 1200px !important;
}
"""

title = """<h1 align="center">AI Extend Image</h1>"""

with gr.Blocks(css=css) as demo:
    with gr.Column():
        gr.HTML(title)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    type="pil",
                    label="Input Image",
                    sources=["upload"],
                    height=300
                )
                
                prompt_input = gr.Textbox(label="Prompt (Optional)", value="high quality")
                
                run_button = gr.Button("Generate", scale=1)

            with gr.Column():
                result = ImageSlider(
                    interactive=False,
                    label="Generated Image",
                )

    run_button.click(
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(
        fn=infer,
        inputs=[input_image, 8, prompt_input],
        outputs=result,
    )

    prompt_input.submit(
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(
        fn=infer,
        inputs=[input_image, 8, prompt_input],
        outputs=result,
    )

demo.queue().launch(share=True, show_error=True, show_api=True, inline=False)
