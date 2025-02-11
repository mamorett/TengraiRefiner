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

# Load environment variables
load_dotenv()

# Enable memory-efficient attention for SD-based models
torch.backends.cuda.enable_mem_efficient_sdp(True)
dtype = torch.bfloat16

class LyingSigmaWrapper(torch.nn.Module):
    def __init__(self, model, dishonesty_factor=-0.05, start_percent=0.1, end_percent=0.9):
        super().__init__()
        self.model = model
        self.dishonesty_factor = dishonesty_factor
        self.start_percent = start_percent
        self.end_percent = end_percent
        
        # Copy essential attributes
        self.dtype = model.dtype
        self.config = model.config

    def forward(self, x, sigma, **kwargs):
        sigma_float = float(sigma.max().detach().cpu())
        
        if self.end_percent <= sigma_float <= self.start_percent:
            sigma = sigma * (1.0 + self.dishonesty_factor)
        
        return self.model(x, sigma, **kwargs)

    def to_empty(self, *args, **kwargs):
        # Ensure that both the base model and the wrapper's parameters are set to empty.
        self.model = self.model.to_empty(*args, **kwargs)
        return super().to_empty(*args, **kwargs)


def create_pipeline(dishonesty_factor=-0.05, start_percent=0.1, end_percent=0.9):
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

    # Wrap the transformer with lying sigma wrapper
    wrapped_transformer = LyingSigmaWrapper(
        transformer,
        dishonesty_factor=dishonesty_factor,
        start_percent=start_percent,
        end_percent=end_percent
    )

    pipe = FluxImg2ImgPipeline(
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=wrapped_transformer,
    )
    
    return pipe

def process_directory(input_dir, dishonesty_factor=-0.05, start_percent=0.1, end_percent=0.9):
    output_dir = os.path.join(input_dir, "FluxSchnell")
    os.makedirs(output_dir, exist_ok=True)

    pipe = create_pipeline(
        dishonesty_factor=dishonesty_factor,
        start_percent=start_percent,
        end_percent=end_percent
    )
    # Get device (e.g., CPU or CUDA) and apply to_empty() with device argument
    device = torch.device("cpu")  # or "cuda" for GPU

    print("Moving models to empty state...")
    pipe.text_encoder.to_empty(device=device)
    pipe.text_encoder_2.to_empty(device=device)
    pipe.vae.to_empty(device=device)
    pipe.transformer.to_empty(device=device)

    # Now, enable CPU offloading after the models are in empty state
    pipe.enable_model_cpu_offload()

    adapter_id = "alimama-creative/FLUX.1-Turbo-Alpha"
    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora(lora_scale=1)

    png_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.png')])
    total_files = len(png_files)

    with tqdm(total=total_files, desc="Processing images", unit="img") as main_pbar:
        for filename in png_files:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            if os.path.exists(output_path):
                main_pbar.update(1)
                continue

            try:
                init_image = Image.open(input_path).convert("RGB")
                width, height = init_image.size
            except Exception as e:
                print(f"Skipping {filename}: {e}")
                main_pbar.update(1)
                continue

            def callback(pipe, step, timestep, callback_kwargs):
                latents = callback_kwargs.get("latents", None)
                callback.step_pbar.update(1)
                print(f"Step {step} / {num_inference_steps} | Timestep: {timestep}")
                return {"latents": latents} if latents is not None else {}

            strength = 0.20
            desired_num_steps = 10
            num_inference_steps = desired_num_steps / strength

            with tqdm(total=desired_num_steps, desc=f"Steps for {filename}", leave=True) as step_pbar:
                callback.step_pbar = step_pbar

                result = pipe(
                    prompt="Very detailed, masterpiece quality",
                    image=init_image,
                    num_inference_steps=num_inference_steps,
                    strength=strength,
                    guidance_scale=3.5,
                    height=height,
                    width=width,
                    callback_on_step_end=callback
                ).images[0]

            result.save(output_path)
            main_pbar.update(1)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python refiner.py <directory_path> [dishonesty_factor] [start_percent] [end_percent]")
    else:
        directory = sys.argv[1]
        dishonesty_factor = float(sys.argv[2]) if len(sys.argv) > 2 else -0.05
        start_percent = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
        end_percent = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9
        
        process_directory(
            directory,
            dishonesty_factor=dishonesty_factor,
            start_percent=start_percent,
            end_percent=end_percent
        )