from PIL import Image
import os



def scaleImage(filepath, scalingDimensions):
    with Image.open(filepath) as img:
        scaledImage = img.resize(scalingDimensions)
    return scaledImage



def scaleImages(rawImageFolderName, outputFolderName):
    fileNames = [filename for filename in os.listdir(f"./{rawImageFolderName}")]
    unscaledImagePaths = [f"{rawImageFolderName}/{filename}" for filename in fileNames]
    scaledImagePaths = [f"{outputFolderName}/{filename}" for filename in fileNames]

    scalingDimensions = (200, 200)
    
    allScalingSuccessful = True

    for filename, inputPath, outputPath in zip(fileNames, unscaledImagePaths, scaledImagePaths):
        try:
            scaledImg = scaleImage(inputPath, scalingDimensions)
            scaledImg.save(outputPath)
            scaledImg.close()
        except Error as e:
            allScalingSuccessful = False
    return allScalingSuccessful



def main():
    # Scaling the training images
    success = scaleImages("training_cropped_images", "training_scaled_images")
    assert success
    
    # Scaling the testing images
    success = scaleImages("testing_cropped_images", "testing_scaled_images")
    assert success
    
    # Scale the colored testing images
    success = scaleImages("training_colored_images", "training_colored_scaled_images")
    assert success
    
    # Scale the colored training images
    success = scaleImages("testing_colored_cropped_images", "testing_colored_scaled_images")
    assert success


if __name__ == "__main__":
    main()



