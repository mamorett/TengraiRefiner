import torchvision
torchvision.disable_beta_transforms_warning()
from dotenv import load_dotenv
import os
import torch
from PIL import Image
from diffusers import FluxImg2ImgPipeline, FluxPriorReduxPipeline, FluxPipeline
from tqdm import tqdm
import datetime
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from optimum.quanto import freeze, qfloat8, quantize
import warnings
from huggingface_hub import hf_hub_download
warnings.filterwarnings('ignore')
import argparse

load_dotenv()

# Enable memory-efficient attention for SD-based models
torch.backends.cuda.enable_mem_efficient_sdp(True)
dtype = torch.bfloat16

bfl_repo = "black-forest-labs/FLUX.1-dev"
# revision = "refs/pr/3"
revision = "main"
repo_redux = "black-forest-labs/FLUX.1-Redux-dev"

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision)
tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, revision=revision)
vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision)
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, revision=revision)


def process_directory(input_dir, output_dir, acceleration, redux, prompt):
    os.makedirs(output_dir, exist_ok=True)
    if redux:
        pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(repo_redux, torch_dtype=dtype)
        pipe = FluxPipeline.from_pretrained(
            bfl_repo, 
            text_encoder=None,
            text_encoder_2=None,
            torch_dtype=dtype
        )      
        # pipe = FluxPriorReduxPipeline(
        #     scheduler=scheduler,
        #     text_encoder=text_encoder,
        #     tokenizer=tokenizer,
        #     text_encoder_2=text_encoder_2,
        #     tokenizer_2=tokenizer_2,
        #     vae=vae,
        #     transformer=transformer,
        # )
    else:
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

    if acceleration == "hyper":
        repo_name = "ByteDance/Hyper-SD"
        ckpt_name = "Hyper-FLUX.1-dev-8steps-lora.safetensors"
        pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        pipe.fuse_lora(lora_scale=0.125)
    elif acceleration == "alimama":
        adapter_id = "alimama-creative/FLUX.1-Turbo-Alpha"
        pipe.load_lora_weights(adapter_id)
        pipe.fuse_lora(lora_scale=1)

    print(datetime.datetime.now(), "Quantizing transformer")
    quantize(transformer, weights=qfloat8)
    freeze(transformer)

    print(datetime.datetime.now(), "Quantizing text encoder 2")
    quantize(text_encoder_2, weights=qfloat8)
    freeze(text_encoder_2)

    # Check if input_dir is a file or directory
    if os.path.isfile(input_dir):
        # If input_dir is a file, extract its directory and filename separately
        input_dir, filename = os.path.split(input_dir)
        png_files = [filename]  # Store only the filename
    elif os.path.isdir(input_dir):
        # If input_dir is a directory, leave the code as it was originally.
        png_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.png')])
    else:
        raise ValueError("Input must be either a file or a directory.")

    # Create a list of files that need processing, excluding already processed files
    files_to_process = []
    print (f"output_dir directory: {output_dir}")
    
    for f in png_files:
        filename = os.path.basename(f)  # Extract only the filename
        output_path = os.path.join(output_dir, filename)  # Output path is based on the filename
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
                if redux:
                    strength = 1.0
                else:
                    strength = 0.20
                if acceleration in ["alimama", "hyper"]:
                    desired_num_steps = 10
                else:
                    desired_num_steps = 25
                # see https://huggingface.co/docs/diffusers/api/pipelines/flux#diffusers.FluxImg2ImgPipeline for more details
                num_inference_steps = desired_num_steps / strength

                if redux:
                    num_images = 1
                    pipe_prior_output = pipe_prior_redux(init_image)
                else:
                    num_images = 1

                with tqdm(total=desired_num_steps, desc=f"Steps for {fname}", leave=True) as step_pbar:
                    callback.step_pbar = step_pbar
                    if redux:
                        result = pipe(
                            guidance_scale=2.5,
                            num_inference_steps=int(num_inference_steps),
                            generator=torch.Generator("cpu").manual_seed(0),
                            **pipe_prior_output,
                        ).images
                    else:
                        result = pipe(
                            prompt=prompt,
                            image=init_image,
                            num_inference_steps=int(num_inference_steps),
                            strength=strength,
                            guidance_scale=3.0,
                            height=height,
                            width=width,
                            num_images_per_prompt=num_images,
                            callback_on_step_end=callback
                        ).images

                    # Saving images with appropriate filenames
                    if len(result) > 1:
                        # If multiple images, add suffixes
                        for idx, img in enumerate(result):
                            output_image_path = f"{output_path.rstrip('.png')}_{str(idx + 1).zfill(4)}.png"  # For example: image_0001.png
                            img.save(output_image_path)
                    else:
                        # If only one image, save normally
                        result[0].save(output_path)

                main_pbar.update(1)                

            except Exception as e:
                print(f"Skipping {fname}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Process PNG files.")
    parser.add_argument('path', type=str, 
                        help='The path of the directory to process')

    parser.add_argument('--acceleration', '-a', type=str,
                        choices=['alimama', 'hyper'],
                        default='alimama',
                        help='Acceleration LORA. Available options are Alimama Turbo or ByteDance Hyper (alimama|hyper) with 10 steps. If not provided, flux with 25 steps will be used.')
    
    parser.add_argument('--prompt', '-p', type=str,
                    default='Very detailed, masterpiece quality',
                    help='Set a custom prompts, if not defined defaults to Very detailed, masterpiece quality')
    
    parser.add_argument('--redux', '-r', action='store_true',
                        help="Use redux instead of img2img")            

    # Create mutually exclusive group
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--output_dir', '-o', type=str,
                       help='Optional output directory. If not provided, outputs will be placed in current directory.')

    group.add_argument('--subdir', '-s', type=str,
                       help='Use subdir output directory. It will save all files in the specified subdirectory of the given path.')

    args = parser.parse_args()

    # Ensure `path` is either a file or directory
    if not os.path.exists(args.path):
        print(f"Error: {args.path} does not exist.")
        exit(1)

    if not args.prompt:
        args.prompt = 'Very detailed, masterpiece quality'

    # Determine output directory
    if args.subdir:
        out_dir = os.path.join(args.path, args.subdir)
    elif args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.getcwd() # Default to current directory

    print (f"Output directory: {out_dir}")

    process_directory(args.path, out_dir, args.acceleration, args.redux, args.prompt)

if __name__ == "__main__":
    main()