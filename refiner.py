import torchvision
from torchvision import transforms
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
from diffusers.utils import load_image

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

class LyingSigmaSampler:
    def __init__(self, 
                 dishonesty_factor: float = -0.05, 
                 start_percent: float = 0.1, 
                 end_percent: float = 0.9):
        self.dishonesty_factor = dishonesty_factor
        self.start_percent = start_percent
        self.end_percent = end_percent

    def __call__(self, model, x, sigmas, **kwargs):
        start_percent, end_percent = self.start_percent, self.end_percent
        ms = model.inner_model.inner_model.model_sampling
        start_sigma, end_sigma = (
            round(ms.percent_to_sigma(start_percent), 4),
            round(ms.percent_to_sigma(end_percent), 4),
        )
        del ms

        def model_wrapper(x, sigma, **extra_args):
            sigma_float = float(sigma.max().detach().cpu())
            if end_sigma <= sigma_float <= start_sigma:
                sigma = sigma * (1.0 + self.dishonesty_factor)
            return model(x, sigma, **extra_args)

        for k in ("inner_model", "sigmas"):
            if hasattr(model, k):
                setattr(model_wrapper, k, getattr(model, k))

        return model_wrapper(x, sigmas, **kwargs)

def process_directory(input_dir, output_dir, acceleration, redux, prompt, fp8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)
    if redux:
        pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
            repo_redux, 
            torch_dtype=dtype
        )    
        pipe_prior_redux.enable_model_cpu_offload()

        pipe = FluxPipeline(
            scheduler=scheduler,
            text_encoder=None,
            tokenizer=tokenizer,
            text_encoder_2=None,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=transformer,
        )
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

    if not fp8:
        print(datetime.datetime.now(), "Quantizing transformer")
        quantize(transformer, weights=qfloat8)
        freeze(transformer)
    else:
        try:
            print(f"Loading FP8 transformer: {fp8}")
            fp8_transformer = torch.load(fp8, weights_only=False, map_location=device)
            fp8_transformer.eval()
            pipe.transformer = fp8_transformer
        except Exception as e:
            print(f"Error loading FP8 transformer: {e}")
            print("Falling back to default transformer")
    if not redux:
        print(datetime.datetime.now(), "Quantizing text encoder")
        quantize(text_encoder, weights=qfloat8)
        freeze(text_encoder)

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
                # init_image = Image.open(input_path).convert("RGB")
                init_image = load_image(input_path)
                width, height = init_image.size
                # Add your image processing logic here
                current_pixels = width * height
                
                # If image is already 1MP or larger, return original
                if current_pixels <= 1_000_000:
                    init_image=upscale_to_sdxl(input_path)
                    width, height = init_image.size       

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

                # Detailer Daemon
                pipe.scheduler.set_sigmas = LyingSigmaSampler(
                    dishonesty_factor=-0.05,
                    start_percent=0.1,
                    end_percent=0.9
                )

                with torch.no_grad():  # Add this context manager
                    if redux:
                        pipe_prior_output = pipe_prior_redux(image=init_image)
                        num_images = 1
                    else:
                        num_images = 1
                    with tqdm(total=desired_num_steps, desc=f"Steps for {fname}", leave=True) as step_pbar:
                        callback.step_pbar = step_pbar
                        if redux:
                            # print("Running redux")
                            result = pipe(
                                guidance_scale=2.5,
                                num_inference_steps=int(num_inference_steps),
                                height=height,
                                width=width,                                
                                # generator=torch.Generator("cpu").manual_seed(0),
                                **pipe_prior_output,
                                callback_on_step_end=callback
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


def upscale_to_sdxl(image_path):
    """
    Upscale image to nearest SDXL resolution (maintaining aspect ratio) if below 1 megapixel.
    Common SDXL resolutions: 1024x1024, 1024x576, 576x1024, 1152x896, 896x1152, etc.
    
    Args:
        image_path (str): Path to input image
    
    Returns:
        PIL.Image: Resized image object
    """
    # Open the image
    img = Image.open(image_path)
    
    # Get current dimensions
    width, height = img.size
    
    # Calculate aspect ratio
    aspect_ratio = width / height
    
    # SDXL base sizes to consider
    sdxl_sizes = [
        (1024, 1024),  # 1:1
        (1024, 576),   # 16:9
        (576, 1024),   # 9:16
        (1152, 896),   # 9:7
        (896, 1152),   # 7:9
        (1024, 768),   # 4:3
        (768, 1024),   # 3:4
        (1216, 832),   # Additional sizes
        (832, 1216),
        (1344, 768),
        (768, 1344),
        (1536, 640),
        (866, 1155),        
        (640, 1536)
    ]
    
    # Filter out sizes that are smaller than 1 megapixel
    sdxl_sizes = [(w, h) for w, h in sdxl_sizes if w * h >= 1000000]
    
    # Find the best matching SDXL resolution
    best_size = None
    min_ratio_diff = float('inf')
    
    for w, h in sdxl_sizes:
        current_ratio = w / h
        ratio_diff = abs(current_ratio - aspect_ratio)
        
        if ratio_diff < min_ratio_diff:
            min_ratio_diff = ratio_diff
            best_size = (w, h)
    
    # Resize image using LANCZOS resampling (high quality)
    resized_img = img.resize(best_size, Image.LANCZOS)   
    return resized_img


def main():
    parser = argparse.ArgumentParser(description="Process PNG files.")
    parser.add_argument('path', type=str, 
                        help='The path of the directory to process')

    parser.add_argument('--acceleration', '-a', type=str,
                        choices=['alimama', 'hyper', 'none'],
                        default='none',
                        help='Acceleration LORA. Available options are Alimama Turbo or ByteDance Hyper (alimama|hyper) with 10 steps. If not provided, flux with 25 steps will be used.')
    
    parser.add_argument('--prompt', '-p', type=str,
                    default='Very detailed, masterpiece quality',
                    help='Set a custom prompts, if not defined defaults to Very detailed, masterpiece quality')
    
    parser.add_argument('--redux', '-r', action='store_true',
                        help="Use redux instead of img2img")           

    parser.add_argument('--load-fp8', '-q', type=str,
                        help="Use a local FP8 quantized transformer model")  

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

    process_directory(args.path, out_dir, args.acceleration, args.redux, args.prompt, args.load_fp8)

if __name__ == "__main__":
    main()