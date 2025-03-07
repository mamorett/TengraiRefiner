<div align="center">
<pre>
██████ ██████ ██   ██  ████  ██████   ██   ██████ ██████ ██████ ██████ ██████ ██   ██ ██████ ██████ 
  ██   ██     ███  ██ ██     ██  ██  ████    ██   ██  ██ ██     ██       ██   ███  ██ ██     ██  ██ 
  ██   ████   ██ █ ██ ██ ███ ██████ ██  ██   ██   ██████ ████   ████     ██   ██ █ ██ ████   ██████ 
  ██   ██     ██  ███ ██  ██ ██ ██  ██████   ██   ██ ██  ██     ██       ██   ██  ███ ██     ██ ██  
  ██   ██████ ██   ██  ████  ██  ██ ██  ██ ██████ ██  ██ ██████ ██     ██████ ██   ██ ██████ ██  ██ 
</pre>
</div>
<p align="center">
	<em><code>❯ REPLACE-ME</code></em>
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
</p>
<br>

##  Table of Contents

- [Table of Contents](#table-of-contents)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Arguments](#arguments)
  - [Options](#options)
  - [Examples](#examples)
- [Processing Details](#processing-details)
- [Memory Optimization](#memory-optimization)
- [Error Handling](#error-handling)
- [Notes](#notes)
- [Project Roadmap](#project-roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---
TengraiRefiner is a Python script for batch processing images using FLUX models with optional acceleration via Alimama Turbo or ByteDance Hyper LORA adapters. It is meant mainly to act as refiner for images produced by Tengrai AI (www.tengrai.ai) but can be obivously used to enhance any image using Flux.dev.

## Features

- Support for both single image and batch processing
- Compatible with FLUX.1-dev and FLUX.1-Redux-dev models
- Memory-efficient processing with automatic CPU offloading
- Quantization support for optimal performance
- Progress tracking with detailed step information
- Configurable acceleration options (Alimama Turbo or ByteDance Hyper)

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

- `-a, --acceleration`: Choose acceleration LORA (options: 'alimama' or 'hyper', default: 'alimama')
- `-p, --prompt`: Set a custom prompt (default: 'Very detailed, masterpiece quality')
- `-r, --redux`: Use redux instead of img2img
- `-o, --output_dir`: Specify output directory (mutually exclusive with --subdir)
- `-s, --subdir`: Save output in a subdirectory of the input path (mutually exclusive with --output_dir)

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

## Processing Details

- Images are processed one at a time with progress tracking
- Default processing uses 25 inference steps (10 steps with acceleration)
- Strength parameter is set to 0.20 for img2img and 1.0 for redux
- Already processed images are skipped to avoid duplication
- Output maintains original image dimensions

## Memory Optimization

The script includes several optimizations:
- Memory-efficient attention for SD-based models
- BFloat16 precision
- Automatic CPU offloading
- Transformer and text encoder quantization
- Model freezing for reduced memory usage

## Error Handling

- Skips already processed images
- Provides error messages for failed processing attempts
- Validates input paths and arguments
- Continues processing remaining images if one fails

## Notes

- Requires CUDA-capable GPU for optimal performance
- Progress bars show both overall progress and per-image steps
- Environment variables can be configured via .env file
- Original file names are preserved in output
- FP8 safetensor files must be compatible with FluxTransformer2DModel

---
##  Project Roadmap

- [X] **`Task 1`**: <strike>Support Refiner mode and Redux mode</strike>
- [ ] **`Task 2`**: Implement memory optimization for Redux.

---

##  Contributing

- **💬 [Join the Discussions](https://github.com/mamorett/TengraiRefiner/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/mamorett/TengraiRefiner/issues)**: Submit bugs found or log feature requests for the `TengraiRefiner` project.
- **💡 [Submit Pull Requests](https://github.com/mamorett/TengraiRefiner/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
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
6. **Push to github**: Push the changes to your forked repository.
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
   <a href="https://github.com{/mamorett/TengraiRefiner/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=mamorett/TengraiRefiner">
   </a>
</p>
</details>

---

##  License

This project is protected under the [GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/) License. For more details, refer to the [LICENSE](./LICENSE) file.

---

##  Acknowledgments



---
