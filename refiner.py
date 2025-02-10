from dotenv import load_dotenv
import os
import torch
from PIL import Image
from diffusers import FluxPipeline
from safetensors.torch import load_file
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import gc  # For manual garbage collection
import datetime
from datetime import timezone
from multiformats import multihash
from optimum.quanto import freeze, qfloat8, quantize
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers import FluxImg2ImgPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

# Load environment variables
load_dotenv()

# Enable memory-efficient attention for SD-based models
torch.backends.cuda.enable_mem_efficient_sdp(True)
dtype = torch.bfloat16

bfl_repo = "black-forest-labs/FLUX.1-dev"
revision = "refs/pr/3"

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision)
tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, revision=revision)
vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision)
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, revision=revision)

print(datetime.datetime.now(), "Quantizing transformer")
quantize(transformer, weights=qfloat8)
freeze(transformer)

print(datetime.datetime.now(), "Quantizing text encoder 2")
quantize(text_encoder_2, weights=qfloat8)
freeze(text_encoder_2)


# # Function to load model weights lazily to reduce memory footprint
# def load_model_weights(model_path):
#     return load_file(model_path)

# Load the Flux model pipeline
def process_directory(input_dir):
    output_dir = os.path.join(input_dir, "FluxSchnell")
    os.makedirs(output_dir, exist_ok=True)
    
    # model_path = os.getenv('FLUX_MODEL_PATH')
    # if not model_path:
    #     raise ValueError("FLUX_MODEL_PATH not set in .env file")
        
    # print(f"Loading Flux model from {model_path}...")
    # pipe = FluxPipeline.from_single_file(model_path).to(device, dtype=dtype)
    pipe = FluxImg2ImgPipeline(
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=transformer,
    )
    pipe.enable_model_cpu_offload()    
    
    # Load LoRA weights and set the LoRA scale to exactly 0.125
    # repo_name = "ByteDance/Hyper-SD"
    # ckpt_name = "Hyper-FLUX.1-dev-8steps-lora.safetensors"
    adapter_id = "alimama-creative/FLUX.1-Turbo-Alpha"

    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora(lora_scale=1)    

    # Process images one by one
    png_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.png')])
    
    for filename in tqdm(png_files, desc="Processing images", unit="img"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Check if the output image already exists
        if os.path.exists(output_path):
            continue
            
        try:
            # Open the input image
            init_image = Image.open(input_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        # Progress callback during image processing
        def callback(step, timestep, latents):
            callback.pbar.update(1)
            return {"latents": latents}
        
        num_inference_steps = 10
        with tqdm(total=num_inference_steps, desc=f"Steps for {filename}", leave=False) as pbar:
            callback.pbar = pbar
            # Perform the image-to-image inference
            result = pipe(
                prompt="Very detailed, masterpiece quality",
                image=init_image,
                num_inference_steps=num_inference_steps,
                strength=0.20,
                guidance_scale=3.5,
            ).images[0]         
        
        # Save the resulting image
        result.save(output_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory_path>")
    else:
        process_directory(sys.argv[1])
