import torchvision
from torchvision import transforms
torchvision.disable_beta_transforms_warning()
from dotenv import load_dotenv
import os
import torch
from PIL import Image
from diffusers import FluxImg2ImgPipeline, FluxPriorReduxPipeline, FluxPipeline, FluxControlPipeline
from tqdm import tqdm
import datetime
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast, AutoModelForCausalLM, AutoProcessor
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from optimum.quanto import freeze, qfloat8, quantize
import warnings
from huggingface_hub import hf_hub_download
warnings.filterwarnings('ignore')
import argparse
from diffusers.utils import load_image
from image_gen_aux import DepthPreprocessor
import time
import torch.profiler
from safetensors.torch import load_file

load_dotenv()

# Enable memory-efficient attention for SD-based models
torch.backends.cuda.enable_mem_efficient_sdp(True)
dtype = torch.bfloat16
bfl_repo = "black-forest-labs/FLUX.1-dev"
# revision = "refs/pr/3"
# revision = "main"
repo_redux = "black-forest-labs/FLUX.1-Redux-dev"
repo_depth = "black-forest-labs/FLUX.1-Depth-dev"

device = "cuda" if torch.cuda.is_available() else "cpu"


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

def prepare_repo(repo_name, revision="main"):
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
    text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision)
    tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, revision=revision)
    vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision)
    transformer = FluxTransformer2DModel.from_pretrained(repo_name, subfolder="transformer", torch_dtype=dtype, revision=revision)
    return scheduler, text_encoder, tokenizer, text_encoder_2, tokenizer_2, vae, transformer


def miaoshuai_tagger(image):
    # Use AutoModelForCausalLM instead of directly loading the model class
    model = AutoModelForCausalLM.from_pretrained(
        "MiaoshouAI/Florence-2-large-PromptGen-v2.0", 
        trust_remote_code=True
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(
        "MiaoshouAI/Florence-2-large-PromptGen-v2.0", 
        trust_remote_code=True
    )

    prompt = "<MORE_DETAILED_CAPTION>"

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=77,
        do_sample=False,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))

    # Extract the string from the dictionary
    if isinstance(parsed_answer, dict) and prompt in parsed_answer:
        result = parsed_answer[prompt]
    else:
        result = str(parsed_answer)  # Fallback to string representation
    
    print(result)
    return result  # Return the string instead of the dictionary


def apply_loras(lorafile, pipe):
    if not lorafile:
        return pipe

    # Apply each LoRA sequentially, ensuring weights are loaded on the GPU
    with torch.device(device):
        if lorafile:
            lora_name = os.path.basename(lorafile)  # Extract only the filename
            print(f"Applying LoRA: {lora_name}")
            try:
                pipe.load_lora_weights(lorafile, weight_name=lora_name, device=device)  # Load LoRA on GPU
                pipe.fuse_lora(lora_scale=1.0)  # Adjust scale as needed
                pipe.unload_lora_weights()  # Clean up immediately
            except Exception as e:
                print(f"Failed to apply LoRA {lora_name}: {str(e)}")
                # Continue with next LoRA even if one fails
    return pipe


def setup_pipeline(mode, acceleration, lora_file, safetensor_path=None):
    """Set up and configure the appropriate pipeline based on parameters."""
    pipe_prior_redux = None

    # Aggressive memory cleanup
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    print(f"Initial VRAM usage (PyTorch): {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    print(f"Total reserved VRAM: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")

    if mode == "depth":
        scheduler, text_encoder, tokenizer, text_encoder_2, tokenizer_2, vae, transformer = prepare_repo(repo_depth, revision="main")
    else:
        scheduler, text_encoder, tokenizer, text_encoder_2, tokenizer_2, vae, transformer = prepare_repo(bfl_repo, revision="main")
    
    # Load safetensor if provided
    if safetensor_path:
        print(f"Loading transformer from safetensor: {safetensor_path}")
        try:
            # Load to CPU and keep it there initially
            state_dict = load_file(safetensor_path, device="cpu")
            transformer.load_state_dict(state_dict, strict=False)
            transformer.eval()
            print("Successfully loaded safetensor weights to CPU")
        except Exception as e:
            print(f"Error loading safetensor: {e}")
            print("Falling back to default transformer")
            torch.cuda.empty_cache()

    # Pipeline creation
    if mode == "redux":
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
    elif mode == "refiner":
        pipe = FluxImg2ImgPipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=transformer,
        )
    else:
        pipe = FluxControlPipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=transformer,
        )

    # Apply offloading based on safetensor usage
    print("Using model CPU offload for quantized mode")
    try:
        pipe.enable_model_cpu_offload()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"CUDA OOM during model offload: {e}")
            print("Falling back to sequential CPU offload")
            pipe.enable_sequential_cpu_offload()
        else:
            raise e

    # Additional optimizations
    try:
        pipe.enable_vae_slicing()
    except AttributeError:
        print("VAE slicing not supported, skipping...")
    try:
        pipe.enable_attention_slicing(4)
    except AttributeError:
        print("Attention slicing not supported, skipping...")
    pipe.set_progress_bar_config(disable=True)
    
    # Clear cache before LoRA
    torch.cuda.empty_cache()
    print(f"VRAM after pipeline setup: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    
    # Apply LoRA
    if lora_file:
        try:
            pipe = apply_loras(lora_file, pipe)
            print(f"VRAM after LoRA: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"CUDA OOM during LoRA: {e}")
                print("Skipping LoRA application")
            else:
                raise e

    # Apply acceleration
    if acceleration == "hyper":
        repo_name = "ByteDance/Hyper-SD"
        ckpt_name = "Hyper-FLUX.1-dev-8steps-lora.safetensors"
        try:
            pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
            pipe.fuse_lora(lora_scale=0.125)
            print(f"Loaded Hyper-SD adapter: {ckpt_name}")
        except RuntimeError as e:
            print(f"Failed to load Hyper-SD: {e}")
    elif acceleration == "alimama":
        adapter_id = "alimama-creative/FLUX.1-Turbo-Alpha"
        try:
            pipe.load_lora_weights(adapter_id)
            pipe.fuse_lora(lora_scale=1)
            print(f"Loaded adapter: {adapter_id}")
        except RuntimeError as e:
            print(f"Failed to load Alimama: {e}")

    # Quantize if no safetensor
    if not safetensor_path:
        print(datetime.datetime.now(), "Quantizing transformer")
        quantize(transformer, weights=qfloat8)
        freeze(transformer)
        print(f"VRAM after quantization: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")

    # Final memory state
    print(f"Final VRAM usage: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    print(f"Peak VRAM usage: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
    torch.cuda.empty_cache()
                
    return pipe, pipe_prior_redux


def prepare_image(input_path, scale_down=False):
    """
    Prepare the input image with appropriate scaling.
    Parameters:
    input_path (str): The file path to the input image.
    scale_down (bool): Flag to indicate whether to scale down the image if it exceeds a certain size. Default is False.
    Returns:
    tuple: A tuple containing the processed image, its width, and its height.
    The function performs the following steps:
    1. Loads the image from the given input path.
    2. If scale_down is True, checks the image size and scales it down if it exceeds 1.5 million pixels.
    3. Ensures the image is at least 1 million pixels by scaling it up if necessary.
    4. Returns the processed image along with its width and height.
    """
    """Prepare the input image with appropriate scaling."""
    init_image = load_image(input_path)
    fname = os.path.basename(input_path)
    
    # Debug scale-down process thoroughly
    print(f"\nProcessing {fname}:")
    # Apply scale down if requested
    if scale_down:
        width, height = init_image.size
        pixel_count = width * height
        print(f"  Original size: {width}x{height} ({pixel_count} pixels)")
        if pixel_count > 1_500_000:  # Between 1.5MP and 2.2MP, scale down 2x
            init_image = upscale_to_sdxl(input_path)
        else:
            print(f"  Image below 1.5MP, no scaling applied")
        width, height = init_image.size  # Update dimensions after resize
        print(f"  Size after resize: {width}x{height} ({width * height} pixels)")
        
    width, height = init_image.size  # Ensure we use the latest dimensions
    current_pixels = width * height                    
    # If image is already 1MP or larger, return original
    if current_pixels <= 1_000_000:
        print(f"  Image below 1Mpixel, scaling up to nearest SDXL resolution")
        init_image = upscale_to_sdxl(input_path)
        width, height = init_image.size       

    print(f"  Final processing size: {width}x{height} ({current_pixels} pixels)")
    
    return init_image, width, height

def warmup_pipeline(pipe, pipe_prior_redux, mode):
    dummy_image = Image.new("RGB", (128, 128), (255, 255, 255))
    if mode == "redux":
        pipe_prior_redux(image=dummy_image)
        pipe(**pipe_prior_redux(image=dummy_image), num_inference_steps=1)
    elif mode == "refiner":
        pipe(prompt="test", image=dummy_image, num_inference_steps=1)
    else:  # depth
        processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf").to(device)
        control_image = processor(dummy_image)[0].convert("RGB")
        pipe(prompt="test", control_image=control_image, num_inference_steps=1)


def process_single_image(input_path, output_path, pipe, pipe_prior_redux, prompt, mode, acceleration, strength, scale_down):
    """
    Process a single image with the configured pipeline.
    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the processed image.
        pipe (Pipeline): The main image processing pipeline.
        pipe_prior_redux (Pipeline): The pipeline used for redux processing.
        prompt (str): The prompt for image generation.
        redux (bool): Flag to indicate if redux processing should be used.
        acceleration (str): Type of acceleration to be used ('alimama', 'hyper', etc.).
        strength (float): Strength parameter for img2img processing.
        scale_down (float): Factor to scale down the input image.
    Returns:
        bool: True if processing was successful, False otherwise.
    """
    try:
        # Prepare the image
        init_image, width, height = prepare_image(input_path, scale_down)
        fname = os.path.basename(input_path)
        
        def callback(pipe, step, timestep, callback_kwargs):
            latents = callback_kwargs.get("latents", None)
            callback.step_pbar.update(1)
            return {"latents": latents} if latents is not None else {}
        
        # Set the timesteps and strength for the inference
        if mode == "redux" or mode == "depth":
            # Redux always uses strength 1.0
            effective_strength = 1.0
        else:
            # Use the provided strength parameter for img2img
            effective_strength = strength
            
        if acceleration in ["alimama", "hyper"]:
            desired_num_steps = 10
        else:
            desired_num_steps = 25

        # Calculate inference steps based on strength
        num_inference_steps = desired_num_steps / effective_strength

        if prompt == "auto":
            prompt = miaoshuai_tagger(init_image)

        # Detailer Daemon
        pipe.scheduler.set_sigmas = LyingSigmaSampler(
            dishonesty_factor=-0.05,
            start_percent=0.1,
            end_percent=0.9
        )

        with torch.no_grad():
            if mode=="redux":
                pipe_prior_output = pipe_prior_redux(image=init_image)
                num_images = 1
            elif mode=="refiner":
                num_images = 1
            else:
                num_images = 1
                processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf").to(device)
                control_image = processor(init_image)[0].convert("RGB")                
                
            with tqdm(total=desired_num_steps, desc=f"Steps for {fname}", leave=True) as step_pbar:
                callback.step_pbar = step_pbar
                if mode == "redux":
                    result = pipe(
                        guidance_scale=2.5,
                        num_inference_steps=int(num_inference_steps),
                        height=height,
                        width=width,                                
                        **pipe_prior_output,
                        callback_on_step_end=callback
                    ).images
                elif mode == "refiner":
                    result = pipe(
                        prompt=prompt,
                        image=init_image,
                        num_inference_steps=int(num_inference_steps),
                        strength=effective_strength,
                        guidance_scale=3.0,
                        height=height,
                        width=width,
                        num_images_per_prompt=num_images,
                        callback_on_step_end=callback
                    ).images
                else:
                    result = pipe(
                        prompt=prompt,
                        control_image=control_image,
                        num_inference_steps=int(num_inference_steps),
                        guidance_scale=10.0,
                        height=height,
                        width=width,
                        num_images_per_prompt=num_images,
                        callback_on_step_end=callback
                    ).images                 

            # Saving images with appropriate filenames
            if len(result) > 1:
                # If multiple images, add suffixes
                for idx, img in enumerate(result):
                    output_image_path = f"{output_path.rstrip('.png')}_{str(idx + 1).zfill(4)}.png"
                    img.save(output_image_path)
            else:
                # If only one image, save normally
                result[0].save(output_path)
                
        return True
    except Exception as e:
        print(f"Error processing {os.path.basename(input_path)}: {e}")
        return False

def get_files_to_process(input_dir, output_dir):
    """
    Get the list of PNG files that need to be processed from the input directory.
    This function checks whether the input path is a file or a directory. If it is a file,
    it extracts the directory and filename separately. If it is a directory, it retrieves
    all PNG files within it. The function then compares the list of PNG files with the
    output directory to determine which files need to be processed (i.e., files that do
    not already exist in the output directory).
    Args:
        input_dir (str): The path to the input directory or file.
        output_dir (str): The path to the output directory.
    Returns:
        tuple: A tuple containing the input directory path and a list of files to process.
    Raises:
        ValueError: If the input path is neither a file nor a directory.
    """
    # Check if input_dir is a file or directory
    if os.path.isfile(input_dir):
        # If input_dir is a file, extract its directory and filename separately
        input_dir_path, filename = os.path.split(input_dir)
        png_files = [filename]  # Store only the filename
        input_dir = input_dir_path  # Update input_dir to be the directory
    elif os.path.isdir(input_dir):
        # If input_dir is a directory, get all image files with supported extensions
        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
        png_files = sorted([f for f in os.listdir(input_dir) 
                        if f.lower().endswith(valid_extensions)])
    else:
        raise ValueError("Input must be either a file or a directory.")

    # Create a list of files that need processing, excluding already processed files
    files_to_process = []
    print(f"Output directory: {output_dir}")

    for f in png_files:
        filename = os.path.basename(f)  # Extract only the filename
        output_path = os.path.join(output_dir, filename)  # Output path is based on the filename
        print(f"Output path: {output_path}")
        if not os.path.exists(output_path):
            files_to_process.append(f)  # Only add files that don't exist in the output directory
        else:
            print(f"Skipping {f}: already exists in output directory.")

    # Debug: print the number of files that need processing
    print(f"Total files to process: {len(files_to_process)}")

    return input_dir, files_to_process


def process_directory(input_dir, output_dir, acceleration, prompt, safetensor_path=None, lora_file=None, scale_down=False, strength=0.20, mode="refiner"):
    """
    Process all images in a directory with the specified parameters.
    Args:
        input_dir (str): The directory containing input images.
        output_dir (str): The directory where processed images will be saved.
        acceleration (bool): Flag to enable or disable acceleration.
        redux (bool): Flag to enable or disable redux mode.
        prompt (str): The prompt to be used for processing images.
        fp8 (bool): Flag to enable or disable FP8 precision.
        lora_file (str): Path to the LoRA file to be used.
        scale_down (bool, optional): Flag to enable or disable scaling down of images. Defaults to False.
        strength (float, optional): The strength of the processing effect. Defaults to 0.20.
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup the pipeline with safetensor_path
    pipe, pipe_prior_redux = setup_pipeline(mode, acceleration, lora_file, safetensor_path)
    
    print("Warming up pipeline...")
    warmup_pipeline(pipe, pipe_prior_redux, mode)
    
    input_dir, files_to_process = get_files_to_process(input_dir, output_dir)
    
    with tqdm(total=len(files_to_process), desc="Processing images", unit="img") as main_pbar:
        for filename in files_to_process:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            success = process_single_image(
                input_path, 
                output_path, 
                pipe, 
                pipe_prior_redux, 
                prompt, 
                mode, 
                acceleration, 
                strength,
                scale_down
            )
            if success:
                main_pbar.update(1)


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

    parser.add_argument('--mode', '-m', type=str,
                    choices=['refiner', 'redux', 'depth'],
                    default='refiner',
                    help='Set mode of operation (refiner|redux|depth), if not defined defaults to refiner')     

    parser.add_argument('--safetensor', '-t', type=str,
                        help="Path to a Flux safetensor file to load transformer weights")

    parser.add_argument('--scale-down', '-s', action='store_true',
                        help="Scale down the source image by 50% before processing if above 1.5 megapixels")

    parser.add_argument('--lora', '-l', type=str,
                        help="Path to a LoRA file to apply before acceleration")

    parser.add_argument('--denoise', '-d', type=float,
                        help="Denoise strength for refine processing")                        

    parser.add_argument('--output_dir', '-o', type=str,
                       help='Optional output directory. If not provided, outputs will be placed in current directory.')

    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: {args.path} does not exist.")
        exit(1)

    if not args.prompt:
        args.prompt = 'Very detailed, masterpiece quality'

    out_dir = args.output_dir if args.output_dir else os.getcwd()

    print(f"Output directory: {out_dir}")

    process_directory(args.path, out_dir, args.acceleration, args.prompt, args.safetensor, args.lora, args.scale_down, args.denoise, args.mode)

if __name__ == "__main__":
    main()