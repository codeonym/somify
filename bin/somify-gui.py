import sys
import os

# Add root absolute path for custom package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image
from PIL import ImageTk
import tkinter as tk
from tkinter import filedialog
import cv2
from ksom.compress_image import compress_image
from ksom.errorLogger import ErrorLogger
from customtkinter import *

def get_entry_value(entry_widget):
    return entry_widget.get()

class SOMApp(CTk):
    def __init__(self):
        super().__init__()
        # Error Handler
        self.error_handler = ErrorLogger()

        # Error Log path
        self.error_path = os.path.join("logs", "errorLogs.txt")
        self.geometry("800x600")
        self.title("SOMify -  Simple Image Compressor")
        # Load and set the logo image

        set_appearance_mode("dark")
        custom_font = ("Helvetica", 14)

        # Left Frame - Vertical split
        self.left_frame = CTkFrame(self)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.image_label = CTkLabel(self.left_frame, text="Select Image", font=custom_font)
        self.image_label.pack(pady=(20, 10))  # Adjust top padding

        self.select_image_button = CTkButton(self.left_frame, text="Browse Image", command=self.browse_image)
        self.select_image_button.pack(pady=10)

        self.block_width_label = CTkLabel(self.left_frame, text="Block Width", font=custom_font)
        self.block_width_label.pack(pady=10)

        self.block_width_entry = CTkComboBox(self.left_frame, values=["Please Select An Image"])
        self.block_width_entry.pack(pady=10)

        self.bits_per_codevector_label = CTkLabel(self.left_frame, text="Bits per Codevector", font=custom_font)
        self.bits_per_codevector_label.pack(pady=10)

        self.bits_per_codevector_entry = CTkEntry(self.left_frame)
        self.bits_per_codevector_entry.pack(pady=10)

        self.epochs_label = CTkLabel(self.left_frame, text="Epochs", font=custom_font)
        self.epochs_label.pack(pady=10)

        self.epochs_entry = CTkEntry(self.left_frame)
        self.epochs_entry.pack(pady=10)

        self.alpha_label = CTkLabel(self.left_frame, text="Alpha", font=custom_font)
        self.alpha_label.pack(pady=10)

        self.alpha_entry = CTkEntry(self.left_frame)
        self.alpha_entry.pack(pady=10)

        self.grayscale_var = tk.BooleanVar()
        self.grayscale_checkbox = CTkCheckBox(self.left_frame, text="Grayscale", variable=self.grayscale_var)
        self.grayscale_checkbox.pack(pady=10)

        self.compress_button = CTkButton(self.left_frame, text="Compress Image", command=self.compress_image)
        self.compress_button.pack(pady=10)

        # Right Frame
        self.right_frame = CTkFrame(self)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.output_image_label = CTkLabel(self.right_frame, text="Output Image", font=custom_font)
        self.output_image_label.pack(pady=(20, 10))  # Adjust top padding

        self.save_image_button = CTkButton(self.right_frame, text="Save Image", command=self.save_image)
        self.save_image_button.pack(pady=10)

        self.mse_label = CTkLabel(self.right_frame, text="MSE Value", font=custom_font)
        self.mse_label.pack(pady=10)

        # Attributes to store the PhotoImage and compressed image
        self.photo = None
        self.compressed_image = None

    def browse_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("All files", "*.*")])
        if file_path and any(file_path.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif")):
            self.image_path = file_path
            original_image = Image.open(file_path)
            image_with = original_image.width
            image_height = original_image.height
            # Resize the image to a reasonable size
            max_display_width = 400
            max_display_height = 400
            original_image.thumbnail((max_display_width, max_display_height))

            # Keep a reference to the PhotoImage object
            self.photo = ImageTk.PhotoImage(original_image)

            # Use CTkButton to display the image on the button
            self.select_image_button.configure(image=self.photo, compound=tk.TOP)
            self.select_image_button.image = self.photo

            # Adjust the button size to fit the resized image
            self.select_image_button.configure(width=original_image.width, height=original_image.height)

            # Calculate the width options
            divisors = [str(i) for i in range(1, image_with) if image_with % i == 0 and image_height % i == 0]
            divisors.append(str(image_with))
            self.block_width_entry.configure(values=divisors)
            self.compress_button.configure(state=tk.NORMAL)

    def compress_image(self):
        block_width = int(get_entry_value(self.block_width_entry))
        bits_per_codevector = int(get_entry_value(self.bits_per_codevector_entry))
        epochs = int(get_entry_value(self.epochs_entry))
        alpha = float(get_entry_value(self.alpha_entry))
        grayscale = self.grayscale_var.get()

        try:
            image = cv2.imread(self.image_path)
            image_width, image_height = image.shape[1], image.shape[0]
            aspect_ratio = image_width / image_height
            block_height = int(block_width / aspect_ratio)

            if block_width > image_width or block_height > image_height:
                raise ValueError(f"Width*height must be inferior to {image_width}*{image_height}")

            if image_width % block_width != 0 or image_height % block_height != 0:
                raise ValueError(f"Inconsistent dimensions {block_width}*{block_height}")

            output_image, mse_value = compress_image(image, block_width, block_height, bits_per_codevector, epochs,
                                                     alpha, grayscale)
            # Convert BGR to RGB if needed
            if output_image.shape[-1] == 3:
                output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

            output_image = Image.fromarray(output_image)
            output_image.thumbnail((400, 400))
            # Keep a reference to the PhotoImage object
            self.photo = ImageTk.PhotoImage(output_image)

            # Keep a reference to the PhotoImage object
            self.photo = ImageTk.PhotoImage(output_image)

            # Use CTkLabel to display the image on the right frame
            self.save_image_button.configure(image=self.photo, compound=tk.TOP)
            self.save_image_button.image = self.photo

            # Adjust the label size to fit the resized image
            self.save_image_button.configure(width=output_image.width, height=output_image.height)

            self.mse_label.configure(text=f"MSE Value: {mse_value:.4f}")

            # Store the compressed image in the instance variable
            self.compressed_image = output_image
        except Exception as e:
            self.mse_label.configure(text=f"Error: Error Occurred Check File: {self.error_path} For More Details")
            self.error_handler.error_log(e.__str__())

    def save_image(self):
        try:
            # Get the file path to save the image
            save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if save_path:
                # Save the compressed image
                self.compressed_image.save(save_path)
                print("Image saved successfully!")
        except Exception as e:
            self.mse_label.configure(text=f"Error: Image Not Saved. Check File: {self.error_path} For More Details")
            self.error_handler.error_log(e.__str__())


if __name__ == "__main__":
    app = SOMApp()
    app.mainloop()
