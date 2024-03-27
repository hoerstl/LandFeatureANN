"""
Code and network framework gathered from:
https://www.tensorflow.org/tutorials/images/cnn

Contributers
- Thao Pham
- Lawrence Hoerst
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle




class TwoWayDict(dict):
    """
    Class and functionality taken from stack overflow:
    https://stackoverflow.com/questions/1456373/two-way-reverse-map#:~:text=class%20TwoWayDict(,__len__(self)%20//%202
    """
    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2





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
    
    
    model = models.Sequential()
    
    # unpickle our test data here
    with open('training_inputs.pickle', 'rb') as inputsFile:
        training_inputs = pickle.load(inputsFile)
    with open('training_outputs.pickle', 'rb') as outputsFile:
        training_outputs = pickle.load(outputsFile)
        
        

    
    
    # We need to decide on how many layers we want
    # These parameters are not yet setup for our network
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    print(model.summary())
    
    model.add(layers.Flatten())
    model.add(layers.Dense(200*200, activation='relu'))
    model.add(layers.Dense(200*200))
    
    
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    # TODO: We need to add our validation inputs/outputs here
    history = model.fit(training_inputs, training_outputs, epochs=10)
    
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)



if __name__ == "__main__":
    main()




