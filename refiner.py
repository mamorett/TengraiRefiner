from dotenv import load_dotenv
import os
import torch
from PIL import Image
from diffusers import FluxImg2ImgPipeline
from tqdm import tqdm
import datetime
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from optimum.quanto import freeze, qfloat8, quantize
from diffusers import EulerDiscreteScheduler
import warnings
warnings.filterwarnings('ignore')
import argparse


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

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

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
    pipe.set_progress_bar_config(disable=True)
    # pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    adapter_id = "alimama-creative/FLUX.1-Turbo-Alpha"
    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora(lora_scale=1)  

    # Get the list of PNG files in the input directory
    png_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.png')])

    # Create a list of files that need processing, excluding already processed files
    files_to_process = []
    for f in png_files:
        output_path = os.path.join(output_dir, f)  # Output path is based on the filename
        print (f"Output path: {output_path}")
        if not os.path.exists(output_path):
            files_to_process.append(f)  # Only add files that don't exist in the output directory
        else:
            print(f"Skipping {f}: already exists in output directory.")  # Debugging line

    # Debug: print the number of files that need processing
    print(f"Total files to process: {len(files_to_process)}")

    total_files_to_process = len(files_to_process)

    with tqdm(total=total_files_to_process, desc="Processing images", unit="img") as main_pbar:
        # Create the list of input files and output file paths for only the files to process
        input_files = [os.path.join(input_dir, filename) for filename in files_to_process]
        output_files = {filename: os.path.join(output_dir, filename) for filename in files_to_process}

        # Process each file
        for input_path in input_files:
            fname = os.path.basename(input_path)
            output_path = output_files[fname]

            try:
                # Process the image file
                init_image = Image.open(input_path).convert("RGB")
                width, height = init_image.size
                # Add your image processing logic here

                def callback(pipe, step, timestep, callback_kwargs):
                    latents = callback_kwargs.get("latents", None)
                    callback.step_pbar.update(1)
                    return {"latents": latents} if latents is not None else {}  # Ensure a valid return type

                # Set the timesteps and strength for the inference
                strength = 0.20
                desired_num_steps = 10
                # see https://huggingface.co/docs/diffusers/api/pipelines/flux#diffusers.FluxImg2ImgPipeline for more details
                num_inference_steps = desired_num_steps / strength

                with tqdm(total=desired_num_steps, desc=f"Steps for {fname}", leave=True) as step_pbar:
                    callback.step_pbar = step_pbar

                    result = pipe(
                        prompt="Very detailed, masterpiece quality",
                        image=init_image,
                        num_inference_steps=int(num_inference_steps),
                        strength=strength,
                        guidance_scale=3.5,
                        height=height,
                        width=width,
                        callback_on_step_end=callback
                    ).images[0]

                result.save(output_path)
                main_pbar.update(1)                

            except Exception as e:
                print(f"Skipping {fname}: {e}")


def main():
    # Custom usage string for better clarity and control over help messages.
    parser = argparse.ArgumentParser(description="Process PNG files.")
    parser.add_argument('directory_path', type=str, 
                        help='The path of the directory to process')

    parser.add_argument('--output_dir', '-o', type=str,
                        help='Optional output directory. If not provided, outputs will be placed in {directory_path}/FluxSchnell.')

    # Parse arguments
    args = parser.parse_args()

    # If output_dir is not provided, set it based on directory_path
    if not args.output_dir:
        args.output_dir = os.path.join(args.directory_path, "FluxSchnell")    
    
    # Check if the mandatory argument is missing
    if not args.directory_path:
        print("Error: Missing required argument 'directory_path'.")
        parser.print_help()  # Print help message with custom usage string.
        return

    # Call your function with arguments
    process_directory(str(args.directory_path), str(args.output_dir))

if __name__ == "__main__":
    main()