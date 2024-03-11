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







def main():
    
    
    
    
    
    classNames = ['water', 'buildings', 'roads', 'foliage', 'mineral deposits', 'mountainous', 'rocky', 'sandy', 'plains', 'snow', 'grass']
    model = models.Sequential()
    
    # We need to decide on how many layers we want
    # These parameters are not yet setup for our network
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    print(model.summary())
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    
    
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))
    
    pass



if __name__ == "__main__":
    main()




