"""
Land Feature ANN - the converter 

Code and network framework gathered from:
https://www.tensorflow.org/tutorials/images/cnn

Contributors:
- Thao Pham
- Lawrence Hoerst
"""

# import tensorflow
import tensorflow as tf 

# import libraries
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
from PIL import ImageColor

# list files in a directory
import os

def images_processing(images_path):

    images_array = []

    # Load images:
    for train_image_name in os.listdir(images_path):

        # concatenate directory paths with filenames or additional directories that is correct for the host operating system
        images_join_path = os.path.join(images_path, train_image_name)

        # open images:
        train_image = Image.open(images_join_path)

        # Resize images:
        train_image = train_image.resize((200,200))

        # Convert images to numpy array:
        # Each pixel will have three values corresponding to the RGB channels:
        train_image_np = np.array(train_image)

        # normalize the pixel values between 0 and 1:
        # train_image_np = train_image_np / 255.0
        images_array.append(train_image_np)
    
    return np.array(images_array)


def main():
    # Class names to plot the images:
    class_names = ['Water', 'Buildings', 'Roads', 'Foliage', 'Mineral deposits', 'Mountainous terrain', 'Rocky terrain', 'Sandy terrain', 'Plains', 'Snow', 'Grass']
    class_colors = []
    class_codes = {class_names[i]: i for i in range(len(class_names)}
    
    # hex_to_rgb
    cvt = lambda hex: ImageColor.getcolor("hex", "RGB")
    colors_to_code = {cvt('#0f5e9c'): 0,
                      cvt('#f2f2f2'): 1, cvt('#606060'): 1,
                      cvt('#c4c4c4'): 2,
                      cvt('#3a5f0b'): 3,
                      cvt('#490e0e'): 4,
                      cvt('#5a7a4c'): 5,
                      cvt('#698287'): 6,
                      cvt('#f7ae64'): 7,
                      cvt('#c89e23'): 8,
                      cvt('#fffafa'): 9,
                      cvt('#7cfc00'): 10}


    

    images_path = r'C:\Users\phamt2\ANNproject\LandFeatureANN\colored_images'
    images_array = images_processing(images_path)

    

    # Display the first 25 images from the training set to verify the data:
    plt.figure(figsize=(10,10))
    for i in range(10):
        plt.subplot(5,5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images_array[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[0]) # not sure yet
    plt.show()

if __name__ == "__main__":
    main()

