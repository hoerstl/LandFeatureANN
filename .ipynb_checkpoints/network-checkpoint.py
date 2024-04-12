"""
Code and network framework gathered from:
https://www.tensorflow.org/tutorials/images/cnn

Contributers
- Thao Pham
- Lawrence Hoerst
"""
import tensorflow as tf
# Maybe I can take out datasets
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt




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
    
    
    
    
    
    classNames = ['water', 'buildings', 'roads', 'foliage', 'mineral deposits', 'mountainous', 'rocky', 'sandy', 'plains', 'snow', 'grass']
    model = models.Sequential()
    
    # TODO: unpickle our test data here
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel classification values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    
    class_names = ['Water', 'Buildings', 'Roads', 'Foliage', 'Mineral deposits', 'Mountainous terrain', 'Rocky terrain', 'Sandy terrain', 'Plains', 'Snow', 'Grass']
    class_codes = {class_names[i]: i for i in range(len(class_names))}
    

    # plt.figure(figsize=(10,10))
    # for i in range(25):
        # plt.subplot(5,5,i+1)
        # plt.xticks([])
        # plt.yticks([])
        # plt.grid(False)
        # plt.imshow(train_images[i])
        # # The CIFAR labels happen to be arrays, 
        # # which is why you need the extra index
        # plt.xlabel(class_names[train_labels[i][0]])
    # plt.show()
    
    #############################################################################
    
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

    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))
    
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)



if __name__ == "__main__":
    main()




