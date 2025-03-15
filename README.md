<div align="center">
<pre>
████████ ███████ ███    ██  ██████  ██████   █████  ██ ██████  ███████ ███████ ██ ███    ██ ███████ ██████  
   ██    ██      ████   ██ ██       ██   ██ ██   ██ ██ ██   ██ ██      ██      ██ ████   ██ ██      ██   ██ 
   ██    █████   ██ ██  ██ ██   ███ ██████  ███████ ██ ██████  █████   █████   ██ ██ ██  ██ █████   ██████  
   ██    ██      ██  ██ ██ ██    ██ ██   ██ ██   ██ ██ ██   ██ ██      ██      ██ ██  ██ ██ ██      ██   ██ 
   ██    ███████ ██   ████  ██████  ██   ██ ██   ██ ██ ██   ██ ███████ ██      ██ ██   ████ ███████ ██   ██ 
</pre>
</div>
<p align="center">
	<em><code>❯ Enhance your images with FLUX.1 models</code></em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/mamorett/TengraiRefiner?style=flat-square&logo=opensourceinitiative&logoColor=white&color=8a2be2" alt="license">
	<img src="https://img.shields.io/github/last-commit/mamorett/TengraiRefiner?style=flat-square&logo=git&logoColor=white&color=8a2be2" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/mamorett/TengraiRefiner?style=flat-square&color=8a2be2" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/mamorett/TengraiRefiner?style=flat-square&color=8a2be2" alt="repo-language-count">
</p>
<p align="center">Built with the tools and technologies:</p>
<p align="center">
	<img src="https://img.shields.io/badge/tqdm-FFC107.svg?style=flat-square&logo=tqdm&logoColor=black" alt="tqdm">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/FLUX-FFC107.svg?style=flat-square&logo=pytorch&logoColor=black" alt="FLUX">
	<img src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white" alt="PyTorch">
	<img src="https://img.shields.io/badge/HuggingFace-FF9A00.svg?style=flat-square&logo=huggingface&logoColor=white" alt="HuggingFace">
</p>
<br>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Arguments](#arguments)
  - [Options](#options)
  - [Examples](#examples)
- [Processing Details](#processing-details)
- [Image Preparation](#image-preparation)
- [Memory Optimization](#memory-optimization)
- [Error Handling](#error-handling)
- [Notes](#notes)
- [Project Roadmap](#project-roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---
TengraiRefiner is a Python script for batch processing images using FLUX models with optional acceleration via Alimama Turbo or ByteDance Hyper LORA adapters. It is meant mainly to act as a refiner for images produced by Tengrai AI (www.tengrai.ai) but can be obviously used to enhance any image using Flux.dev models.

## Features

- Support for both single image and batch processing
- Compatible with FLUX.1-dev and FLUX.1-Redux-dev models
- Intelligent image scaling to SDXL-compatible resolutions
- Memory-efficient processing with automatic CPU offloading
- FP8 quantization support for optimal performance
- Detailed progress tracking with per-step information
- Configurable acceleration options (Alimama Turbo or ByteDance Hyper)
- Custom LoRA support with ability to apply before acceleration
- Adjustable denoise strength for refiner processing
- Option to scale down large images for more efficient processing

## Prerequisites

Before running the script, ensure you have Python 3.x installed and the following dependencies:

```txt
torch>=2.6.0
diffusers==0.32.2
transformers>=4.35.0
safetensors>=0.4.0
python-dotenv>=1.0.0
Pillow>=10.0.0
tqdm>=4.66.0
huggingface-hub>=0.19.0
optimum-quanto
multiformats
xformers>=0.0.25
```

## Installation

1. Clone this repository or download the script
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The script can be run from the command line with various options:

```bash
python script.py <path> [options]
```

### Arguments

- `path`: Required. Path to input file or directory containing PNG files to process

### Options

- `-a, --acceleration`: Choose acceleration LORA (options: 'alimama', 'hyper', or 'none', default: 'none')
- `-p, --prompt`: Set a custom prompt (default: 'Very detailed, masterpiece quality')
- `-r, --redux`: Use redux instead of img2img
- `-q, --load-fp8`: Use a local FP8 quantized transformer model
- `-s, --scale-down`: Scale down the source image by 50% if above 1.5 megapixels
- `-l, --lora`: Path to a LoRA file to apply before acceleration
- `-d, --denoise`: Denoise strength for refine processing (default: 0.20)
- `-o, --output_dir`: Specify output directory

### Examples

1. Process a single image with default settings:
   ```bash
   python script.py path/to/image.png
   ```

2. Process a directory of images with Hyper acceleration:
   ```bash
   python script.py path/to/directory -a hyper
   ```

3. Process images with a custom prompt and specific output directory:
   ```bash
   python script.py path/to/directory -p "high quality, detailed" -o output/folder
   ```

4. Use redux processing with Alimama acceleration:
   ```bash
   python script.py path/to/directory -r -a alimama
   ```

5. Apply a custom LoRA and set denoise strength:
   ```bash
   python script.py path/to/directory -l path/to/lora.safetensors -d 0.30
   ```

## Processing Details

- Images are processed one at a time with comprehensive progress tracking
- Default processing uses 25 inference steps (10 steps with acceleration)
- Strength parameter is configurable via the `-d/--denoise` option (defaults to 0.20 for img2img and always 1.0 for redux)
- Already processed images are skipped to avoid duplication
- Custom LyingSigmaSampler for improved detail enhancement
- Output maintains original image dimensions

## Image Preparation

The script includes intelligent image preparation:
- Images smaller than 1 megapixel are automatically upscaled to the nearest SDXL resolution
- Images larger than 1.5 megapixels can be scaled down (with the `-s/--scale-down` option)
- Aspect ratio is preserved during scaling
- Compatible with common SDXL resolutions (1024×1024, 1024×576, etc.)

## Memory Optimization

The script includes several optimizations:
- Memory-efficient attention for SD-based models
- BFloat16 precision by default
- Automatic CPU offloading for components not actively in use
- Transformer quantization via optimum-quanto for reduced VRAM usage
- Option to load pre-quantized FP8 transformer models
- Model freezing for reduced memory footprint

## Error Handling

- Skips already processed images
- Provides detailed error messages for failed processing attempts
- Validates input paths and arguments
- Continues processing remaining images if one fails
- Comprehensive debug output for troubleshooting

## Notes

- Requires CUDA-capable GPU for optimal performance
- Progress bars show both overall batch progress and per-image step progress
- Environment variables can be configured via .env file
- Original file names are preserved in output
- For very large images, consider using the `-s/--scale-down` option

---
## Project Roadmap

- [X] **`Task 1`**: <strike>Support Refiner mode and Redux mode</strike>
- [X] **`Task 2`**: <strike>Add support for custom LoRA files</strike>
- [X] **`Task 3`**: <strike>Implement intelligent image scaling for optimal processing</strike>
- [X] **`Task 4`**: <strike>Add FP8 quantization support</strike>
- [ ] **`Task 5`**: Implement memory optimization for Redux
- [ ] **`Task 6`**: Add batch size control for more efficient processing

---

## Contributing

- **💬 [Join the Discussions](https://github.com/mamorett/TengraiRefiner/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/mamorett/TengraiRefiner/issues)**: Submit bugs found or log feature requests for the `TengraiRefiner` project.
- **💡 [Submit Pull Requests](https://github.com/mamorett/TengraiRefiner/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your GitHub account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/mamorett/TengraiRefiner
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to GitHub**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com/mamorett/TengraiRefiner/graphs/contributors">
      <img src="https://contrib.rocks/image?repo=mamorett/TengraiRefiner">
   </a>
</p>
</details>

---

## License

This project is protected under the [GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/) License. For more details, refer to the [LICENSE](./LICENSE) file.

---

## Acknowledgments

- [Black Forest Labs](https://flux.dev) for the FLUX.1 models
- [Alimama Creative](https://huggingface.co/alimama-creative) for the FLUX.1-Turbo adapter
- [ByteDance](https://huggingface.co/ByteDance) for the Hyper-SD acceleration adapter
- [Tengrai AI](https://www.tengrai.ai) for the inspiration behind this refiner

---