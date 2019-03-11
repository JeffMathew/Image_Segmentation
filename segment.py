import sys
import cv2
import numpy

def neighbourhood(image, x, y):

    neighbour_region_numbers = {}
    for i in range(-1, 2):                  #check 8 connected neighbours
        for j in range(-1, 2):
            if (i == 0 and j == 0):
                continue
            if (x+i < 0 or y+j < 0):
                continue
            if (x+i >= image.shape[0] or y+j >= image.shape[1]):
                continue
            if (neighbour_region_numbers.get(image[x+i][y+j]) == None):
                neighbour_region_numbers[image[x+i][y+j]] = 1
            else:
                neighbour_region_numbers[image[x+i][y+j]] += 1


    if (neighbour_region_numbers.get(0) != None):
        del neighbour_region_numbers[0]


    keys = list(neighbour_region_numbers)


    keys.sort()

    if (keys[0] == -1):
        if (len(keys) == 1):                    #New Region
            return -1
        elif (len(keys) == 2):                  # Part of existing region
            return keys[1]
        else:                                   # Part of watershed boundary
            return 0
    else:
        if (len(keys) == 1):                    # Part of existing region
            return keys[0]
        else:                                   #Part of  Watershed boundary
            return 0

def watershed_segmentation(image):
    # list of pixel intensities along with their coordinates
    intensity_list = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):

            intensity_list.append((image[x][y], (x, y)))

    # Sort the list with respect to their pixel intensities, in ascending order
    intensity_list.sort()

    # Create an empty array initialized to -1's of same dimension as input
    segmented_image = numpy.full(image.shape, -1, dtype=int)

    # Iterate the intensity_list in ascending order and update the new matrix
    region_number = 0
    for i in range(len(intensity_list)):
        # Print iteration number in terminal for clarity
        sys.stdout.write("\rPixel {} of {}...".format(i, len(intensity_list)))
        sys.stdout.flush()

        # Get the pixel intensity and the x,y coordinates
        intensity = intensity_list[i][0]
        x = intensity_list[i][1][0]
        y = intensity_list[i][1][1]

        # Get the region number of the current pixel's region by checking its neighbouring pixels
        region_status = neighbourhood(segmented_image, x, y)

        # Assign region number (or) watershed accordingly, at pixel (x, y) of the segmented image
        if (region_status == -1):                   # New region
            region_number += 1
            segmented_image[x][y] = region_number
        elif (region_status == 0):                  # Watershed boundary
            segmented_image[x][y] = 0
        else:                                       # Part of existing region
            segmented_image[x][y] = region_status

    # Return the segmented image
    return segmented_image


def main(argv):
    # Read the input image in grayscale
    img = cv2.imread(argv[0], 0)



        #PRE-Processing (use appropriate thresholding method)
    ret, thresh1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    '''
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(img, (5, 5), 0)

    #thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    #th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #blur = cv2.GaussianBlur(img, (4, 4), 0)
    #ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
     # noise removal
    #kernel = numpy.ones((3,3),numpy.uint8)
    #opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    #ret, thresh1 = cv2.threshold(img, 145, 255, cv2.THRESH_BINARY_INV)
    #blur = cv2.GaussianBlur(thresh, (5, 5), 0)
    #ret, thresh1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    #ret, thresh1 = cv2.threshold(img, 110, 255, cv2.THRESH_TOZERO)'''


    # Perform segmentation using watershed_segmentation on the input image
    segmented_image = watershed_segmentation(thresh1)

    # Save the segmented image
    '''cv2.imwrite("images/target.png", img)'''
    cv2.imwrite("images/target.png", segmented_image)



    # Show the segmented image and original image side by side
    input_image = cv2.resize(img, (0,0), None, 0.3, 0.3)
    seg_image = cv2.resize(cv2.imread("images/target.png", 0), (0,0), None, 0.3, 0.3)
    numpy_horiz = numpy.hstack((input_image, seg_image))
    cv2.imshow('Input image ------------------------ Segmented image', numpy_horiz)
    #cv2.imshow('Thresh Output',seg_image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main(sys.argv[1:])
