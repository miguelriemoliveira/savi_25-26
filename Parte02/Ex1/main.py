import cv2  # import the opencv library


# main function, where our code should be
def main():
    print("python main function")

    image = cv2.imread('lake.jpg', cv2.IMREAD_COLOR)

    # Print info on the image
    print(image.dtype)
    print(type(image))  # the type of the image is numpy.ndarray\
    print('shape = ' + str(image.shape))

    # Get the color of the topleft most pixel
    color = image[28, 22]
    print('Color of the topleft most pixel: ' + str(color))

    # # Create a black rectangle on the left side of the image
    # image[0:30, 0:70] = 0  # zero is color black

    # How to split color channels
    B, G, R = cv2.split(image)

    print('B image shape = ' + str(B.shape))
    cv2.imshow('Original image', image)
    cv2.imshow('Blue channel', B)

    # Try to segment the blue color of the sky
    mask = B > 220
    print('Mask shape = ' + str(mask.shape))
    print('Mask dtype = ' + str(mask.dtype))

    mask = mask.astype('uint8') * 255
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)  # 0 means wait forever until a key is pressed


if __name__ == '__main__':
    main()
