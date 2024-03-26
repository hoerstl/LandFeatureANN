from PIL import Image
import os
import getopt, sys



def convertToPNG(folderPath):
    """A function that replaces every image in a folder with a png conversion"""
    fileNames = os.listdir(folderPath)
    count = 0
    for filename in fileNames:
        if filename.split('.')[1] not in {'jpg', 'jpeg'}: continue
        count += 1
        # load the original image
        non_png_filepath = os.path.join(folderPath, filename)
        with Image.open(non_png_filepath) as non_png_img:
            # convert the original image
            png_img = non_png_img.convert("RGBA")
            new_img_name = filename.split('.')[0] + '.png'
        # save the new image in the same directory
        png_img.save(os.path.join(folderPath, new_img_name))
        # delete the old image
        os.remove(non_png_filepath)
    return count
                

def main():
    system_arguments = sys.argv[1:]
    
    options = 'd:'
    long_options = ['directory=']
    
    try: 
        arguments, values = getopt.getopt(system_arguments, options, long_options)
        for arg, val in arguments:
            if arg in ("-d", "--directory"):
                imgConversionCount = convertToPNG(val)
                print(f"successfully converted all {imgConversionCount} images to png")
        
    except getopt.error as error:
        print(str(error))





if __name__ == '__main__':
    main()









