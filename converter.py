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
import pickle

# list files in a directory
import os

def sort_image_names(nameList):
    # We need to sort this list twice since one sort gives that, for example '59.png' < '6.png' when we want to see 6 first.
    # Therefore, we need to sort first by numbering and then by length of the file to get the 1, 2, 3, 4, ... behavior we want to see
    return sorted(sorted(nameList), key=lambda e: len(e))


def images_processing(images_path):
    global colors_to_code
    images_array = []

    # Load images:
    sorted_image_names = sort_image_names(os.listdir(images_path))
    for train_image_name in sorted_image_names:

        # concatenate directory paths with filenames or additional directories that is correct for the host operating system
        images_join_path = os.path.join(images_path, train_image_name)

        # open images:
        train_image = Image.open(images_join_path)

        # Resize images:
        train_image = train_image.resize((200,200))
        
        # Convert images to numpy array:
        # Each pixel will have three values corresponding to the RGB channels:
        train_image_np = np.array(train_image)
        train_codified_image = np.zeros((len(train_image_np), len(train_image_np[0]), 1))
        for i in range(len(train_image_np)):
            for j in range(len(train_image_np[0])):
                if tuple(train_image_np[i][j]) in colors_to_code:
                    train_codified_image[i][j] = colors_to_code[tuple(train_image_np[i][j])]
                else:
                    train_codified_image[i][j] = colors_to_code['default']

        # normalize the pixel values between 0 and 1:
        # train_image_np = train_image_np / 255.0
        images_array.append(train_codified_image)
    
    
    
    return np.array(images_array)


def main():
    global class_names, class_colors, class_codes, colors_to_code
    # Class names to plot the images:
    class_names = ['Water', 'Buildings', 'Roads', 'Foliage', 'Mineral deposits', 'Mountainous terrain', 'Rocky terrain', 'Sandy terrain', 'Plains', 'Snow', 'Grass']
    class_colors = ['#0f5e9c', ('#f2f2f2', '#606060'), '#c4c4c4', '#3a5f0b', '#490e0e', '#5a7a4c', '#698287', '#f7ae64', '#c89e23', '#fffafa', '#7cfc00']
    class_codes = {class_names[i]: i for i in range(len(class_names))}
    
    # hex_to_rgb
    cvt = lambda hex: ImageColor.getcolor(hex, "RGB")
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
                      cvt('#7cfc00'): 10,
                      'default':      11}


    

    colored_image_path = os.path.join(os.getcwd(), 'colored_images')
    colored_image_array = images_processing(colored_image_path)

    with open('training_outputs.pickle', 'wb') as training_file:
        pickle.dump(colored_image_array, training_file)
    print('Colored images converted successfully...')
    
    
    
    colored_image_names = sort_image_names(os.listdir(colored_image_path))
    raw_image_path = os.path.join(os.getcwd(), 'scaled')
    colored_image_array = images_processing(raw_image_path)
    
    raw_image_array = []
    for image_name in colored_image_names:  # for every colored image we loaded, load the raw image in the same order
        img_path = os.path.join(raw_image_path, image_name)
        train_image = Image.open(img_path)
        train_image = train_image.resize((200,200))
        train_image_np = np.array(train_image)
        raw_image_array.append(train_image_np)
        
    
    with open('training_inputs.pickle', 'wb') as training_file:
        pickle.dump(raw_image_array, training_file)
    print('Raw images converted successfully...')
    

if __name__ == "__main__":
    main()

