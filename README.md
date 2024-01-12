# SOMify - Simple Image Compressor using Self-Organizing Maps (SOM)

![SOMify Logo](assets/logo.png)

## Overview

SOMify is a command-line tool for image compression based on the Self-Organizing Maps (SOM) algorithm. It allows users to compress images by specifying various parameters such as bits per codevector, block width, number of epochs, initial learning rate, and whether to convert the image to grayscale.

## Features

- Image compression using SOM algorithm.
- Customizable parameters for compression.
- Grayscale conversion option.
- 
## Cloning the Repository

```bash
git clone https://github.com/codeonym/somify.git
cd somify
```

## Installation

To use SOMify, you need to have Python installed. Then, you can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Command-Line Interface (CLI)

```bash
python bin/somify.py -i IMAGE_PATH -b BITS_PER_CODEVECTOR -w BLOCK_WIDTH [-e EPOCHS] [-a ALPHA] [-g]
```

- `-i, --image_path`: Path to the input image.
- `-b, --bits_per_codevector`: Number of bits per codevector.
- `-w, --block_width`: Width of the image blocks.
- `-e, --epochs` (optional): Number of epochs for training the SOM (default: 10).
- `-a, --alpha` (optional): Initial learning rate for the SOM (default: 0.3).
- `-g, --grayscale` (optional): Convert the image to grayscale before compression.

### Example

The output will be saved at folder ```output``` named ```output/SOMify-6-bird-red.png```
```bash
python bin/somify.py -i samples/bird-red.png -b 6 -w 1 -e 5 -a 0.7
```

### Graphical Interface (GUI)

Run the command

```bash
python bin/somify-gui.py
```

## Contributing

If you find any issues or have suggestions for improvements, please feel free to contribute! Create an issue or submit a pull request.

