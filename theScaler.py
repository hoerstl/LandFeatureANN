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

    for filename, inputPath, outputPath in zip(fileNames, unscaledImagePaths, scaledImagePaths):
        print(f"Scaling img '{filename}' to size {scalingDimensions}. ", end="")
        scaledImg = scaleImage(inputPath, scalingDimensions)
        scaledImg.save(outputPath)
        print(f"Save successful")
        scaledImg.close()



def main():
    # Scaling the training images
    scaleImages("training_cropped_images", "training_scaled_images")
    
    # Scaling the testing images
    scaleImages("testing_cropped_images", "testing_scaled_images")
    
    # Scale the colored testing images
    scaleImages("training_colored_images", "training_colored_scaled_images")
    
    # Scale the colored training images
    scaleImages("testing_colored_cropped_images", "testing_colored_scaled_images")


if __name__ == "__main__":
    main()



