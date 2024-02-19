from PIL import Image
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='input_sets', help='path to input_folder')
    return parser.parse_args()

def resize_images(input_folder, size=(64, 64)):
    for root, dirs, files in os.walk(input_folder):
        for filename in files:  
            
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff')):
                input_path = os.path.join(root, filename)
                new_root = root.replace(input_folder, input_folder + '_64x64')
                filename_without_extension, file_extension = os.path.splitext(filename)
                output_path = new_root + '/' + filename_without_extension + '.jpg'
                if not os.path.exists(new_root):
                    os.makedirs(new_root)

                # Open the image using Pillow
                with Image.open(input_path) as img:
                    # Find the center of the image
                    center_x, center_y = img.size[0] // 2, img.size[1] // 2
                    if center_x != center_y:
                        # Calculate the size of the square to crop
                        size_square = min(img.size[0], img.size[1])

                        # Calculate the coordinates for cropping the square
                        left = center_x - size_square // 2
                        top = center_y - size_square // 2
                        right = center_x + size_square // 2
                        bottom = center_y + size_square // 2

                        # Crop the image to the calculated square
                        img = img.crop((left, top, right, bottom))
                        
                    # Resize the image to the specified size
                    resized_img = img.resize(size)

                    # Save the resized image in JPG format
                    resized_img.save(output_path, "JPEG")

if __name__ == "__main__":
    # Specify the input and output folders
    args = parse_args()
    
    # Resize images in the input folder and save them to the output folder
    resize_images(args.input_path)
