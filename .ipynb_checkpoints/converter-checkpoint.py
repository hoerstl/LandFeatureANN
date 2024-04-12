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
        train_image_np = train_image_np / 255.0
        images_array.append(train_image_np)
    
    return np.array(images_array)


def main():
    images_path = r'C:\Users\phamt2\ANNproject\LandFeatureANN\colored_images'
    images_array = images_processing(images_path)

    # Class names to plot the images:
    class_names = ['Water', 'Buildings', 'Roads', 'Foliage', 'Mineral deposits', 'Mountainous terrain', 'Rocky terrain', 'Sandy terrain', 'Plains', 'Snow', 'Grass']

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

