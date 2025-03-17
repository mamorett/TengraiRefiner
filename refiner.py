import torchvision
torchvision.disable_beta_transforms_warning()
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
import torch.profiler
from safetensors.torch import load_file
from faker import Faker


# Enable memory-efficient attention for SD-based models
torch.backends.cuda.enable_mem_efficient_sdp(True)
dtype = torch.bfloat16
bfl_repo = "black-forest-labs/FLUX.1-dev"
# revision = "refs/pr/3"
# revision = "main"
repo_redux = "black-forest-labs/FLUX.1-Redux-dev"
repo_depth = "black-forest-labs/FLUX.1-Depth-dev"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Memory optimization techniques
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class LyingSigmaSampler:
    """
    A class to modify the sigma values during model sampling by applying a dishonesty factor.

    Attributes:
        dishonesty_factor (float): The factor by which to modify the sigma values. Default is -0.05.
        start_percent (float): The starting percentage of the sigma range. Default is 0.1.
        end_percent (float): The ending percentage of the sigma range. Default is 0.9.

    Methods:
        __call__(model, x, sigmas, **kwargs):
            Modifies the sigma values during model sampling based on the dishonesty factor.
    """
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


def cleanup():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def prepare_repo(repo_name, revision="main", safetensor_path=None):
    """
    Prepares and loads various models and tokenizers from pretrained repositories.

    Args:
        repo_name (str): The name of the repository to load the transformer model from.
        revision (str, optional): The specific revision of the models to load. Defaults to "main".
        safetensor_path (str, optional): Path to a safetensor file to load the transformer state dict from. Defaults to None.

    Returns:
        tuple: A tuple containing the following elements:
            - scheduler: The loaded FlowMatchEulerDiscreteScheduler.
            - text_encoder: The loaded CLIPTextModel.
            - tokenizer: The loaded CLIPTokenizer.
            - text_encoder_2: The loaded T5EncoderModel.
            - tokenizer_2: The loaded T5TokenizerFast.
            - vae: The loaded AutoencoderKL.
            - transformer: The loaded FluxTransformer2DModel.
    """
    with torch.device("cpu"):
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, revision=revision)
    transformer = FluxTransformer2DModel.from_pretrained(repo_name, subfolder="transformer", torch_dtype=dtype, revision=revision)
    if safetensor_path:
        print("Loading transformer...")
        state_dict = load_file(safetensor_path, device=device)
        transformer.load_state_dict(state_dict, strict=False)
        transformer.eval()
    vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision).to(device)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
    return scheduler, text_encoder, tokenizer, text_encoder_2, tokenizer_2, vae, transformer

def generate_docker_name():
        fake = Faker()
        return f"{fake.word().lower()}_{fake.word().lower()}" 

def miaoshuai_tagger(image):
    """
    Generates a detailed caption for the given image using a pre-trained model.
    Args:
        image (PIL.Image or torch.Tensor): The input image for which the caption is to be generated.
    Returns:
        str: The generated detailed caption for the input image.
    Notes:
        - The function uses the "MiaoshouAI/Florence-2-large-PromptGen-v2.0" model and processor.
        - The model and processor are loaded with `trust_remote_code=True`.
        - The prompt used for generating the caption is "<MORE_DETAILED_CAPTION>".
        - The function processes the input image and prompt, generates the caption, and post-processes the generated text.
        - The result is printed and returned as a string.
    """
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
    """
    Apply a LoRA (Low-Rank Adaptation) to the given pipeline.

    This function loads and applies a LoRA from the specified file to the provided pipeline.
    The LoRA weights are loaded onto the GPU, applied, and then unloaded to free up resources.
    If the LoRA application fails, the function will print an error message and continue with the next LoRA.

    Args:
        lorafile (str): The file path to the LoRA file. If None or empty, the function returns the original pipeline.
        pipe (object): The pipeline object to which the LoRA will be applied. This object must have methods 
                       `load_lora_weights`, `fuse_lora`, and `unload_lora_weights`.

    Returns:
        object: The pipeline object with the applied LoRA.
    """
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
    cleanup()                
    return pipe


def setup_pipeline(mode, acceleration, lora_file, safetensor_path=None):
    """
    Sets up the pipeline for various modes and applies optimizations.
    Args:
        mode (str): The mode of the pipeline. Options are "depth", "redux", "refiner", "rejoy", or other.
        acceleration (str): The type of acceleration to apply. Options are "hyper" or "alimama".
        lora_file (str): Path to the LoRA file to apply.
        safetensor_path (str, optional): Path to the safetensor file. Defaults to None.
    Returns:
        tuple: A tuple containing the main pipeline and the prior redux pipeline (if applicable).
    Raises:
        RuntimeError: If there is an error during LoRA application or loading acceleration adapters.
    """
    pipe_prior_redux = None

    # Aggressive memory cleanup
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    print(f"Initial VRAM usage (PyTorch): {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    print(f"Total reserved VRAM: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")

    if mode == "depth":
        scheduler, text_encoder, tokenizer, text_encoder_2, tokenizer_2, vae, transformer = prepare_repo(repo_depth, revision="main", safetensor_path=safetensor_path)
    else:
        scheduler, text_encoder, tokenizer, text_encoder_2, tokenizer_2, vae, transformer = prepare_repo(bfl_repo, revision="main", safetensor_path=safetensor_path)
    
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
    elif mode == "rejoy":
        pipe = FluxPipeline(
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

    if safetensor_path:
        print("Falling back to sequential CPU offload")
        pipe.enable_sequential_cpu_offload()
    else:
        # Apply offloading based on safetensor usage
        print("Using model CPU offload for quantized mode")
        pipe.enable_model_cpu_offload()

    # Additional optimizations
    try:
        pipe.enable_vae_tiling()
    except AttributeError:
        print("VAE tiling not supported, skipping...")    
    try:
        pipe.enable_vae_slicing()
    except AttributeError:
        print("VAE slicing not supported, skipping...")
    try:
        pipe.enable_attention_slicing(4)
    except AttributeError:
        print("Attention slicing not supported, skipping...")
    pipe.set_progress_bar_config(disable=True)

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
    Prepares an image for further processing by optionally scaling it down and ensuring it meets 
    certain size criteria.
    Args:
        input_path (str): The file path to the input image.
        scale_down (bool, optional): If True, the image will be scaled down if it exceeds a certain 
                                     pixel count. Defaults to False.
    Returns:
        tuple: A tuple containing the processed image, its width, and its height.
    """
    # Load the image
    init_image = load_image(input_path)
    fname = os.path.basename(input_path)
    
    # Convert RGBA to RGB if necessary
    if init_image.mode == 'RGBA':
        print(f"  Converting {fname} from RGBA to RGB")
        init_image = init_image.convert('RGB')
    
    # Debug scale-down process thoroughly
    print(f"\nProcessing {fname}:")
    # Apply scale down if requested
    if scale_down:
        width, height = init_image.size
        pixel_count = width * height
        print(f"  Original size: {width}x{height} ({pixel_count} pixels)")
        if pixel_count > 1_500_000:  # Between 1.5MP and 2.2MP, scale down 2x
            init_image = upscale_to_sdxl(init_image)  # Pass the image object
        else:
            print(f"  Image below 1.5MP, no scaling applied")
        width, height = init_image.size  # Update dimensions after resize
        print(f"  Size after resize: {width}x{height} ({width * height} pixels)")
        print(f"  Image mode after scale-down: {init_image.mode}")
        
    width, height = init_image.size  # Ensure we use the latest dimensions
    current_pixels = width * height                    
    # If image is below 1MP, scale up to nearest SDXL resolution
    if current_pixels <= 1_000_000:
        print(f"  Image below 1Mpixel, scaling up to nearest SDXL resolution")
        init_image = upscale_to_sdxl(init_image)  # Pass the image object
        width, height = init_image.size
        print(f"  Image mode after scale-up: {init_image.mode}")

    print(f"  Final processing size: {width}x{height} ({width * height} pixels)")
    
    return init_image, width, height

def warmup_pipeline(pipe, pipe_prior_redux, mode):
    dummy_image = Image.new("RGB", (128, 128), (255, 255, 255))
    if mode == "redux":
        pipe_prior_redux(image=dummy_image)
        pipe(**pipe_prior_redux(image=dummy_image), num_inference_steps=1)
    elif mode == "refiner":
        pipe(prompt="test", image=dummy_image, num_inference_steps=1)
    elif mode == "rejoy":
        pipe(prompt="test", num_inference_steps=1)        
    else:  # depth
        processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf").to(device)
        control_image = processor(dummy_image)[0].convert("RGB")
        pipe(prompt="test", control_image=control_image, num_inference_steps=1)


def process_single_image(input_path, output_path, pipe, pipe_prior_redux, prompt, mode, acceleration, strength, scale_down, cfg, steps):
    """
    Processes a single image using the specified pipeline and parameters.
    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the processed image.
        pipe (Pipeline): The main pipeline used for image processing.
        pipe_prior_redux (Pipeline): The pipeline used for prior reduction in 'redux' mode.
        prompt (str): The prompt to guide the image generation. If "auto", it will be generated automatically.
        mode (str): The mode of processing. Options are "refiner", "redux", "rejoy", or others.
        acceleration (str): The acceleration mode. Options are "alimama", "hyper", or others.
        strength (float): The strength of the effect applied to the image.
        scale_down (float): The factor by which to scale down the image.
        cfg (float): The guidance scale for the pipeline.
        steps (int): The number of steps for the inference process.
    Returns:
        bool: True if the image was processed successfully, False otherwise.
    """
    try:
        # Prepare the image
        init_image, width, height = prepare_image(input_path, scale_down)
        fname = os.path.basename(input_path)
        
        # If output_path already has a random name (from process_directory), use it
        # Otherwise use the base name from input
        if not output_path.endswith('.png'):  # Indicates it wasn't pre-set with random name
            base_name = os.path.splitext(fname)[0]
            output_path = os.path.join(os.path.dirname(output_path), f"{base_name}.png")
        
        def callback(pipe, step, timestep, callback_kwargs):
            latents = callback_kwargs.get("latents", None)
            callback.step_pbar.update(1)
            return {"latents": latents} if latents is not None else {}
        
        # Set the timesteps and strength for the inference
        if mode == "refiner":
            effective_strength = strength
        else:
            effective_strength = 1.0
            
        if acceleration in ["alimama", "hyper"]:
            desired_num_steps = 10
        else:
            desired_num_steps = steps

        num_inference_steps = desired_num_steps / effective_strength

        if prompt == "auto":
            prompt = miaoshuai_tagger(init_image)

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
            elif mode=="rejoy":
                num_images = 1                
            else:
                num_images = 1
                processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf").to(device)
                control_image = processor(init_image)[0].convert("RGB")                
                
            with tqdm(total=desired_num_steps, desc=f"Steps for {fname}", leave=True) as step_pbar:
                callback.step_pbar = step_pbar
                if mode == "redux":
                    result = pipe(
                        guidance_scale=cfg,
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
                        guidance_scale=cfg,
                        height=height,
                        width=width,
                        num_images_per_prompt=num_images,
                        callback_on_step_end=callback
                    ).images
                elif mode == "rejoy":
                    result = pipe(
                        prompt=prompt,
                        num_inference_steps=int(num_inference_steps),
                        guidance_scale=cfg,
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

            # Saving images with appropriate filenames (always PNG)
            if len(result) > 1:
                for idx, img in enumerate(result):
                    multi_output_path = f"{output_path.rstrip('.png')}_{str(idx + 1).zfill(4)}.png"
                    img.save(multi_output_path)
            else:
                result[0].save(output_path)

        return True
    except Exception as e:
        print(f"Error processing {os.path.basename(input_path)}: {e}")
        return False
    

def get_files_to_process(input_dir, output_dir, random_names=False):
    """
    Get a list of files to process from the input directory or file.
    Parameters:
    input_dir (str): Path to the input directory or file.
    output_dir (str): Path to the output directory.
    random_names (bool): If True, process all files and use random names for output. Default is False.
    Returns:
    tuple: A tuple containing the input directory path and a list of files to process.
    Raises:
    ValueError: If the input is neither a file nor a directory.
    Notes:
    - If `input_dir` is a file, only that file will be processed.
    - If `input_dir` is a directory, all files with valid extensions ('.png', '.jpg', '.jpeg', '.webp') will be considered.
    - If `random_names` is False, files that already exist in the output directory will be skipped.
    - If `random_names` is True, all files will be processed regardless of existing files in the output directory.
    """
    if os.path.isfile(input_dir):
        input_dir_path, filename = os.path.split(input_dir)
        files = [filename]
        input_dir = input_dir_path
    elif os.path.isdir(input_dir):
        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
        files = sorted([f for f in os.listdir(input_dir) 
                       if f.lower().endswith(valid_extensions)])
    else:
        raise ValueError("Input must be either a file or a directory.")

    files_to_process = []
    print(f"Output directory: {output_dir}")

    if not random_names:
        for f in files:
            base_name = os.path.splitext(f)[0]
            output_path = os.path.join(output_dir, f"{base_name}.png")
            
            if not os.path.exists(output_path):
                files_to_process.append(f)
            else:
                print(f"Skipping {f}: {base_name}.png already exists in output directory.")
    else:
        # When using random names, process all files since names will be unique
        files_to_process = files
        print("Using random names - processing all files")

    print(f"Total files to process: {len(files_to_process)}")
    return input_dir, files_to_process


def process_directory(input_dir, output_dir, acceleration, prompt, safetensor_path=None, lora_file=None, scale_down=False, strength=0.35, mode="refiner", cfg=3.0, steps=25, random_names=False):
    """
    Processes all images in the input directory using the specified pipeline and saves the results to the output directory.
    Args:
        input_dir (str): Path to the directory containing input images.
        output_dir (str): Path to the directory where processed images will be saved.
        acceleration (bool): Flag to enable acceleration in the pipeline.
        prompt (str): Text prompt to guide the image processing.
        safetensor_path (str, optional): Path to the safetensor file. Defaults to None.
        lora_file (str, optional): Path to the LoRA file. Defaults to None.
        scale_down (bool, optional): Flag to enable scaling down of images. Defaults to False.
        strength (float, optional): Strength of the processing effect. Defaults to 0.35.
        mode (str, optional): Mode of the pipeline. Defaults to "refiner".
        cfg (float, optional): Configuration parameter for the pipeline. Defaults to 3.0.
        steps (int, optional): Number of steps for the processing. Defaults to 25.
        random_names (bool, optional): Flag to enable random naming of output files. Defaults to False.
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    pipe, pipe_prior_redux = setup_pipeline(mode, acceleration, lora_file, safetensor_path)
    
    print("Warming up pipeline...")
    warmup_pipeline(pipe, pipe_prior_redux, mode)
    
    input_dir, files_to_process = get_files_to_process(input_dir, output_dir, random_names)
    
    with tqdm(total=len(files_to_process), desc="Processing images", unit="img") as main_pbar:
        for filename in files_to_process:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename if not random_names else generate_docker_name() + ".png")
            
            success = process_single_image(
                input_path, 
                output_path, 
                pipe, 
                pipe_prior_redux, 
                prompt, 
                mode, 
                acceleration, 
                strength,
                scale_down,
                cfg,
                steps
            )
            if success:
                main_pbar.update(1)


def upscale_to_sdxl(image):
    """
    Upscale image to nearest SDXL resolution (maintaining aspect ratio) if below 1 megapixel.
    Common SDXL resolutions: 1024x1024, 1024x576, 576x1024, 1152x896, 896x1152, etc.
    
    Args:
        image (PIL.Image): Input image object
    
    Returns:
        PIL.Image: Resized image object
    """   
    # Get current dimensions
    width, height = image.size
    
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
    resized_image = image.resize(best_size, Image.LANCZOS)   
    return resized_image

def main():
    parser = argparse.ArgumentParser(description="Process PNG files.")
    parser.add_argument('path', type=str, 
                        help='The path of the directory to process')

    parser.add_argument('--acceleration', '-a', type=str,
                        choices=['alimama', 'hyper', 'none'],
                        default='none',
                        help='Acceleration LORA. Available options are Alimama Turbo or ByteDance Hyper (alimama|hyper) with 10 steps.')
    
    parser.add_argument('--prompt', '-p', type=str,
                    default='Very detailed, masterpiece quality',
                    help='Set a custom prompts, if not defined defaults to Very detailed, masterpiece quality')

    parser.add_argument('--mode', '-m', type=str,
                    choices=['refiner', 'redux', 'depth', 'rejoy'],
                    default='refiner',
                    help='Set mode of operation (refiner|redux|depth), if not defined defaults to refiner')     

    parser.add_argument('--safetensor', '-t', type=str,
                        help="Path to a Flux safetensor file to load transformer weights")

    parser.add_argument('--scale-down', '-s', action='store_true',
                        help="Scale down the source image by 50% before processing if above 1.5 megapixels")

    parser.add_argument('--lora', '-l', type=str,
                        help="Path to a LoRA file to apply before acceleration")

    parser.add_argument('--denoise', '-d', type=float,
                        default=0.35,  # Add default value
                        help="Denoise strength for refine processing (0.0-1.0). Defaults to 0.35 for refiner mode.")
    
    parser.add_argument('--cfg', '-c', type=float,
                        default=3.0,  # Add default value
                        help="Cfg strength for refine processing")

    parser.add_argument('--steps', '-e', type=int,
                        default=25,  # Add default value
                        help="Number of steps for processing. If not provided and accelerator is not set, flux with 25 steps will be used.")         

    parser.add_argument('--output_dir', '-o', type=str,
                       help='Optional output directory. If not provided, outputs will be placed in current directory.')
    
    parser.add_argument('--random-names', '-r', action='store_true',
                            help="Override default naming behavior and use random docker-style names for output files")    

    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: {args.path} does not exist.")
        exit(1)

    if not args.prompt:
        args.prompt = 'Very detailed, masterpiece quality'

    out_dir = args.output_dir if args.output_dir else os.getcwd()

    print(f"Output directory: {out_dir}")
    print(f"Mode of Operation: {args.mode}")
    print(f"Using random names: {args.random_names}")

    process_directory(args.path, out_dir, args.acceleration, args.prompt, args.safetensor, 
                     args.lora, args.scale_down, args.denoise, args.mode, args.cfg, args.steps, 
                     args.random_names)

if __name__ == "__main__":
    main()