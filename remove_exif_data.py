from PIL import Image, ImageOps
import os

# Path to the folder containing JPEG files
input_folder = "labelstudio-export/images"

# Loop through each JPEG file in the folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
        file_path = os.path.join(input_folder, filename)

        with Image.open(file_path) as img:
            # Rotate the image according to the EXIF orientation data
            img = ImageOps.exif_transpose(img)
            
            # When saving, do not specify EXIF data, so it won't be included
            img.save(file_path, "JPEG")

print("Images have been rotated according to EXIF data and saved without EXIF metadata.")
