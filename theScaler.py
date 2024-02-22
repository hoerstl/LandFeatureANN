from PIL import Image
import os



def scaleImage(filepath, scalingDimensions):
    with Image.open(filepath) as img:
        scaledImage = img.resize(scalingDimensions)
    return scaledImage




def main():
    fileNames = [filename for filename in os.listdir("./cropped")]
    unscaledImagePaths = [f"cropped/{filename}" for filename in fileNames]
    scaledImagePaths = [f"scaled/{filename}" for filename in fileNames]

    scalingDimensions = (200, 200)

    for filename, inputPath, outputPath in zip(fileNames, unscaledImagePaths, scaledImagePaths):
        print(f"Scaling img '{filename}' to size {scalingDimensions}. ", end="")
        scaledImg = scaleImage(inputPath, scalingDimensions)
        # use img.save(filepath) to save an image
        scaledImg.save(outputPath)
        print(f"Save successful")
        scaledImg.close()


if __name__ == "__main__":
    main()



