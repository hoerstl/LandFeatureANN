"""
Land Feature ANN - the converter 

Code and network framework gathered from:
https://www.tensorflow.org/tutorials/images/cnn

Contributors:
- Thao Pham
- Lawrence Hoerst
"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
from PIL import ImageColor
import pickle

# list files in a directory
import os

# Get the image dimensions
import theScaler

def sort_image_names(nameList):
    # We need to sort this list twice since one sort gives that, for example '59.png' < '6.png' when we want to see 6 first.
    # Therefore, we need to sort first by numbering and then by length of the file to get the 1, 2, 3, 4, ... behavior we want to see
    return sorted(sorted(nameList), key=lambda e: len(e))
    
    
    
def encode_image(colorized_img):
    """
    A function which takes in an image's numpy array representation and returns a numpy representation
    where all pixels have been classified according to their color.
    colorized_img: A three-dimensional numpy array where the third dimension is rgb values
    """
    global colors_to_code, class_names
    codified_img = np.zeros((len(colorized_img), len(colorized_img[0]), 1))
    for i in range(len(colorized_img)):
        for j in range(len(colorized_img[0])):
            if tuple(colorized_img[i][j]) in colors_to_code:
                codified_img[i][j] = colors_to_code[tuple(colorized_img[i][j])] + .5
            else:
                codified_img[i][j] = colors_to_code['default'] + .5
    # normalize the pixel values between 0 and 1:
    codified_img = codified_img / (len(class_names))
    return codified_img


def decode_image(encoded_img):
    """
    A function which takes in an encoded image's numpy array representation and returns a numpy
    three-dimensional array which can be converted back to an image
    encoded_img: A two-dimensional array where each pixel is a classification value between 0
    and 1.
    """
    global code_to_colors, class_names
    decoded_image = np.zeros((len(encoded_img), len(encoded_img[0]), 4))
    for i in range(len(encoded_img)):
        for j in range(len(encoded_img[0])):
            classification = int(encoded_img[i][j][0] * (len(class_names)))
            if (classification == 12): classification = 11
            elif (classification > len(class_names)): raise ValueError(f"Somehow, we decoded a pixel to a classification of {classification} which is too large")
            decoded_pixel = code_to_colors[classification]
            for k in range(len(decoded_pixel)):
                decoded_image[i][j][k] = int(decoded_pixel[k])
    return np.uint8(decoded_image)


def images_processing(images_path):
    
    images_array = []

    # Load images:
    sorted_image_names = sort_image_names(os.listdir(images_path))
    for train_image_name in sorted_image_names:

        # concatenate directory paths with filenames or additional directories that is correct for the host operating system
        images_join_path = os.path.join(images_path, train_image_name)

        # open images:
        train_image = Image.open(images_join_path)
        
        # Convert images to numpy array:
        # Each pixel will have three values corresponding to the RGB channels:
        train_image_np = np.array(train_image)
        train_codified_image = encode_image(train_image_np)

        
        images_array.append(train_codified_image)
    
    
    
    return np.array(images_array)



def test_encode_decode(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGBA")
        img = img.resize((200,200))
        img_np = np.array(img)
        
    start_img = img_np
    encoded_img = encode_image(start_img)
    end_img = decode_image(encoded_img)
    return start_img, end_img


def main():
    global class_names, class_colors, class_codes, colors_to_code, code_to_colors
    # Class names to plot the images:
    # old class_names = ['Water', 'Buildings', 'Roads', 'Foliage', 'Mineral deposits', 'Mountainous terrain', 'Rocky terrain', 'Sandy terrain', 'Plains', 'Snow', 'Grass']
    class_names = ['Buildings', 'Roads', 'Default', 'Water', 'Foliage', 'Grass', 'Plains', 'Sandy terrain', 'Rocky terrain', 'Mountainous terrain', 'Mineral deposits', 'Snow']
    # old class_colors = ['#0f5e9c', ('#f2f2f2', '#606060'), '#c4c4c4', '#3a5f0b', '#490e0e', '#5a7a4c', '#698287', '#f7ae64', '#c89e23', '#fffafa', '#7cfc00']
    class_colors = [('#f2f2f2ff', '#606060ff'), '#c4c4c4ff', '#00000000', '#0f5e9cff', '#3a5f0bff', '#7cfc00ff', '#c89e23ff', '#f7ae64ff', '#698287ff', '#5a7a4cff', '#490e0eff', '#fffafaff']
    class_codes = {class_names[i]: i for i in range(len(class_names))}
    
    # hex_to_rgb
    cvt = lambda hex: ImageColor.getcolor(hex, "RGBA")
    colors_to_code = {cvt('#f2f2f2ff'): 0, cvt('#606060ff'): 0,
                      cvt('#c4c4c4ff'): 1,
                      'default': 2,
                      cvt('#0f5e9cff'): 3,
                      cvt('#3a5f0bff'): 4,
                      cvt('#7cfc00ff'): 5,
                      cvt('#c89e23ff'): 6,
                      cvt('#f7ae64ff'): 7,
                      cvt('#698287ff'): 8,
                      cvt('#5a7a4cff'): 9,
                      cvt('#490e0eff'): 10,
                      cvt('#fffafaff'): 11}
    code_to_colors = {value: key for key, value in colors_to_code.items() if key != 'default'}
    code_to_colors[2] = cvt('#00000000')
    # currently working to debug the way that default is added to the code_to_colors dictionary


    
    # start, end = test_encode_decode(os.path.join(os.getcwd(), 'colored_images', '60.png'))
    # print(f"start dimensions: {start.shape} and end dimensions: {end.shape}")
    # beginning_img, end_img = Image.fromarray(start), Image.fromarray(end)
    # print("showing start image")
    # beginning_img.show()
    # print("showing end image")
    # end_img.show()

    print('starting the conversion process')
    
    # Convert the scaled colored training images into nparrays and store them
    # TRAINING OUTPUTS
    training_colored_images_path = os.path.join(os.getcwd(), 'training_colored_scaled_images')
    sorted_training_colored_image_names = sort_image_names(os.listdir(training_colored_images_path))
    training_colored_images_array = images_processing(training_colored_images_path)
    with open('training_outputs.pickle', 'wb') as training_file:
        pickle.dump(training_colored_images_array, training_file)
    print(training_colored_images_array[0].shape)
    print('Training colored images converted successfully...')
    
    # VALIDATION OUTPUTS
    validation_colored_images_path = os.path.join(os.getcwd(), 'testing_colored_scaled_images')
    sorted_validation_colored_image_names = sort_image_names(os.listdir(validation_colored_images_path))
    validation_colored_images_array = images_processing(validation_colored_images_path)
    with open('validation_outputs.pickle', 'wb') as validation_file:
        pickle.dump(validation_colored_images_array, validation_file)
    print(validation_colored_images_array[0].shape)
    print('Validation colored images converted successfully...')
    
    
    # TRAINING INPUTS
    training_input_images_path = os.path.join(os.getcwd(), 'training_scaled_images')    
    training_input_image_array = []
    for image_name in sorted_training_colored_image_names:  # for every colored image we loaded, load the raw image in the same order
        img_path = os.path.join(training_input_images_path, image_name)
        train_image = Image.open(img_path)
        train_image_np = np.array(train_image) / 255
        training_input_image_array.append(train_image_np)
        
    
    with open('training_inputs.pickle', 'wb') as training_file:
        pickle.dump(np.array(training_input_image_array), training_file)
    print(training_input_image_array[0].shape)
    print('Training input images converted successfully...')
    
    
    # VALIDATION INPUTS
    validation_input_images_path = os.path.join(os.getcwd(), 'testing_scaled_images')
    validation_input_image_array = []
    for image_name in sorted_validation_colored_image_names:  # for every colored image we loaded, load the raw image in the same order
        img_path = os.path.join(validation_input_images_path, image_name)
        validation_image = Image.open(img_path)
        validation_image_np = np.array(validation_image) / 255
        validation_input_image_array.append(validation_image_np)
        
    
    with open('validation_inputs.pickle', 'wb') as testing_file:
        pickle.dump(np.array(validation_input_image_array), testing_file)
    print(validation_input_image_array[0].shape)
    print('Validation input images converted successfully...')
    print('Finished all conversions')
    

if __name__ == "__main__":
    main()

