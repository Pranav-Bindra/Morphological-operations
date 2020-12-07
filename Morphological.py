##CLOSING
import numpy as np
from PIL import Image
import cv2
import math
img  = cv2.imread("C:/Users/Gaurav/Desktop/ipmv project/test_bw.png")
struct=  np.array(([1, 1, 1], [1, 1, 1], [1, 1, 1]), dtype='uint8')
ret,img2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,comp_img2=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)

def dialation(img, struct):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    padded_gray = cv2.copyMakeBorder(gray, math.floor(struct.shape[0]/2),  math.floor(struct.shape[0]/2),  math.floor(struct.shape[1]/2),  math.floor(struct.shape[1]/2), cv2.BORDER_CONSTANT, value=[0])
    _, threshgray = cv2.threshold(padded_gray, 127, 1, cv2.THRESH_BINARY)

    (m, n) = threshgray.shape
    (m1, n1) = struct.shape
    
    x1 = np.zeros((m, n))
    
    for i in range(math.floor(m1/2), m - math.ceil(m1/2)):
        for j in range(math.floor(n1/2), n - math.ceil(n1/2)):
            z = np.zeros((m1,n1))
            for k in range((-1*math.floor(m1/2)), math.floor(m1/2)):
                for l in range((-1*math.floor(n1/2)), math.floor(n1/2)):
                    z[(math.floor((m1/2)) + k), (math.floor((n1/2)) + l)] = threshgray[i+k, j+l] * struct[(math.floor((m1/2)) + k), (math.floor((n1/2)) + l)];
                
            if np.array_equal(np.zeros((m1, n1)), z):
                x1[i, j] = 0
            else:
                x1[i, j] = 255
                
    return x1

b = np.array(([1, 1, 1], [1, 1, 1], [1, 1, 1]), dtype='uint8')
def erosion_user_defined(img,kernel):
#Convert image into Binary
    ret,img2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    #erosion = cv2.erode(img2, b)
    #Pad the Original image with zeros
    c = np.pad(img2, pad_width=1, mode='constant', constant_values=0)

    row, col = img2.shape
    d = np.zeros((row, col))

    for i in range(2, row-1):
        for j in range(2, col-1):
            a = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            a[0][0] = c[i-1][j-1]
            a[0][1] = c[i-1][j]
            a[0][2] = c[i-1][j+1]
            a[1][0] = c[i][j-1]
            a[1][1] = c[i][j]
            a[1][2] = c[i][j+1]
            a[2][0] = c[i+1][j-1]
            a[2][1] = c[i+1][j]
            a[2][2] = c[i+1][j+1]
            pqr = np.asarray(a)
            if pqr.all() == b.all():
                d[i][j] = 255
    return d
dialated_image = dialation(img,struct)
closing_output = erosion_user_defined(dialated_image, b)
close_2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, b)

cv2.imshow("output",closing_output)
cv2.imshow("op",close_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
##DILATION
import numpy as np
#from pil import Image
import cv2
import math

img = cv2.imread('photo.tif',1)
struct = np.ones((3, 3))
ret,img2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,comp_img2=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)


def dialation(img, struct):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    padded_gray = cv2.copyMakeBorder(gray, math.floor(struct.shape[0] / 2), math.floor(struct.shape[0] / 2),
                                     math.floor(struct.shape[1] / 2), math.floor(struct.shape[1] / 2),
                                     cv2.BORDER_CONSTANT, value=[0])
    _, threshgray = cv2.threshold(padded_gray, 127, 1, cv2.THRESH_BINARY)

    (m, n) = threshgray.shape
    (m1, n1) = struct.shape

    x1 = np.zeros((m, n))

    for i in range(math.floor(m1 / 2), m - math.ceil(m1 / 2)):
        for j in range(math.floor(n1 / 2), n - math.ceil(n1 / 2)):
            z = np.zeros((m1, n1))
            for k in range((-1 * math.floor(m1 / 2)), math.floor(m1 / 2)):
                for l in range((-1 * math.floor(n1 / 2)), math.floor(n1 / 2)):
                    z[(math.floor((m1 / 2)) + k), (math.floor((n1 / 2)) + l)] = threshgray[i + k, j + l] * struct[
                        (math.floor((m1 / 2)) + k), (math.floor((n1 / 2)) + l)];

            if np.array_equal(np.zeros((m1, n1)), z):
                x1[i, j] = 0
            else:
                x1[i, j] = 255

    return x1


eroded = dialation(img, struct)
cv2.imshow('Dilation', eroded)
dialate1=cv2.dilate(img2, struct)
cv2.imshow('without fn',dialate1)
cv2.waitKey(0)
cv2.destroyAllWindows()

##HITANDMISS
import cv2
import numpy as np

#Input the image
img1= cv2.imread('fingerprint.tif', 0)

#binary
ret,img2 = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)
ret,comp_img2=cv2.threshold(img1,127,255,cv2.THRESH_BINARY_INV)

#Kernel
b1 = np.array(([1, 1, 1], [0, 0, 0], [0, 0, 0]), dtype='uint8')
b2 = np.array(([0, 0, 0], [1, 1, 1], [1, 1, 1]), dtype='uint8')

#with fn output image
a1=cv2.morphologyEx(img2,cv2.MORPH_HITMISS,b1)

#1st image
#pad array for original image
c = np.pad(img2, pad_width=1, mode='constant', constant_values=0)

row, col = img2.shape
d1 = np.zeros((row, col))

for i in range(2, row-1):
        for j in range(2, col-1):
            a = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            a[0][0] = c[i-1][j-1]
            a[0][1] = c[i-1][j]
            a[0][2] = c[i-1][j+1]
            a[1][0] = c[i][j-1]
            a[1][1] = c[i][j]
            a[1][2] = c[i][j+1]
            a[2][0] = c[i+1][j-1]
            a[2][1] = c[i+1][j]
            a[2][2] = c[i+1][j+1]
            pqr = np.asarray(a)
            if pqr.all() == b1.all():
                d1[i][j] = 255

#2nd image
#pad array for comp. image
c2 = np.pad(comp_img2, pad_width=1, mode='constant', constant_values=0)

row, col = comp_img2.shape
d2 = np.zeros((row, col))

for i in range(2, row-1):
        for j in range(2, col-1):
            a = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            a[0][0] = c[i-1][j-1]
            a[0][1] = c[i-1][j]
            a[0][2] = c[i-1][j+1]
            a[1][0] = c[i][j-1]
            a[1][1] = c[i][j]
            a[1][2] = c[i][j+1]
            a[2][0] = c[i+1][j-1]
            a[2][1] = c[i+1][j]
            a[2][2] = c[i+1][j+1]
            pqr = np.asarray(a)
            if pqr.all() == b2.all():
                d2[i][j] = 255


z=np.multiply(d1,d2)
a2=255-a1

cv2.imshow('Original Image',img1)
cv2.imshow('Binary Image',img2)
cv2.imshow('Erosion Using User Defined Function',z )
cv2.imshow('Erosion Using Inbuilt Function', a2)
cv2.waitKey(0)
cv2.destroyAllWindows()
##EROSION
import cv2
import numpy as np

#Input the image
img1 = cv2.imread('photo1.tif', 0)
#Kernel
b = np.array(([1, 1, 1], [1, 1, 1], [1, 1, 1]), dtype='uint8')



def erosion_inbuilt(img,kernel):
    ret,img2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    erosion_1 = cv2.erode(img2, b)
    return erosion_1

def erosion_user_defined(img,kernel):
#Convert image into Binary
    ret,img2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    #erosion = cv2.erode(img2, b)
    #Pad the Original image with zeros
    c = np.pad(img2, pad_width=1, mode='constant', constant_values=0)

    row, col = img2.shape
    d = np.zeros((row, col))

    for i in range(2, row-1):
        for j in range(2, col-1):
            a = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            a[0][0] = c[i-1][j-1]
            a[0][1] = c[i-1][j]
            a[0][2] = c[i-1][j+1]
            a[1][0] = c[i][j-1]
            a[1][1] = c[i][j]
            a[1][2] = c[i][j+1]
            a[2][0] = c[i+1][j-1]
            a[2][1] = c[i+1][j]
            a[2][2] = c[i+1][j+1]
            pqr = np.asarray(a)
            if pqr.all() == b.all():
                d[i][j] = 255
    return d


d = erosion_user_defined(img1,b)
d1 = erosion_inbuilt(img1,b)
cv2.imshow('Erosion Using User Defined Function', d)
cv2.imshow('Erosion Using Inbuilt Function', d1)
cv2.waitKey(0)
cv2.destroyAllWindows()




