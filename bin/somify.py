import sys
import os
#  Add root absolute path for custom package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# For image handling
import cv2
# For arguments parsing
import argparse
# For image Compression
from ksom.compress_image import compress_image
# For logo Rendering
from ksom.logoHandler import LogoHandler
from ksom.errorLogger import ErrorLogger


# Define the main function
def main():

    # INIT
    error_handler = ErrorLogger()
    error_path = os.path.join("logs", "errorLogs.txt")
    logo_handler = LogoHandler()

    # Create an argument parser
    parser = argparse.ArgumentParser(description="SOMify Image Compressor")

    # Add the required arguments
    parser.add_argument("-i", "--image_path", help="Path to the input image", required=True)
    parser.add_argument("-b", "--bits_per_codevector",
                        type=int, help="Number of bits per codevector", required=True)
    parser.add_argument("-w", "--block_width", type=int,
                        help="Width of the image blocks", required=True)

    # Add the optional arguments
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="Number of epochs for training the SOM (default: 10)")
    parser.add_argument("-a", "--alpha", type=float, default=0.3,
                        help="Initial learning rate for the SOM (default: 0.3)")
    parser.add_argument("-g", "--grayscale", action="store_true",
                        help="Convert the image to grayscale before compression")

    # Parse the arguments
    args = parser.parse_args()

    # Print The Logo ASCII
    logo_handler.print_ascii_art()

    try:
        # Validate the arguments
        if args.bits_per_codevector < 1 or args.bits_per_codevector > 24:
            raise ValueError("Bits per codevector must be between 1 and 24")
        if args.block_width < 1:
            raise ValueError("Block width must be positive")
        if args.epochs < 1:
            raise ValueError("Epochs must be positive")
        if args.alpha <= 0 or args.alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")

        # Read the image from the given path
        image = cv2.imread(args.image_path)

        # Image dimensions
        image_width, image_height = image.shape[1], image.shape[0]

        # Calculate the aspect ratio
        aspect_ratio = image_width / image_height

        # Calculate block height based on aspect ratio and block width
        block_height = int(args.block_width / aspect_ratio)

        # Check width and height validation
        if args.block_width > image_width or block_height > image_height:
            raise ValueError(f"Width*height must be inferior to {image_width}*{image_height}")

        # Check dimensions validation
        if image_width % args.block_width != 0 or image_height % block_height != 0:
            raise ValueError(f"Inconsistent dimensions {args.block_width}*{block_height}")

        # Compress the image using the SOM algorithm
        output_image, mse = compress_image(image, args.block_width, block_height,
                                           args.bits_per_codevector,
                                           args.epochs, args.alpha, args.grayscale)

        # Generate the output image name
        output_image_name = os.path.join(os.path.dirname(__file__), "..", "output",
                                         f'SOMify-{args.block_width}-{os.path.basename(args.image_path)}')

        # Save the output image
        cv2.imwrite(output_image_name, output_image)

        # Print the MSE between the original and the compressed image
        print("Mean Square Error = ", mse)
    except Exception as e:

        # Log To Standard Error Stream
        sys.stderr.write(f'Error: {e.__str__()}')

        # Log To File Stream
        error_handler.error_log(e)


# Call the main function
if __name__ == "__main__":
    main()
