#coding=utf-8  
import cv2  
import numpy  
  
image = cv2.imread("input1.jpg",0);

element = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))
kernel = element
dilate = cv2.dilate(image, element)  
erode = cv2.erode(image, element)  
  

result = cv2.absdiff(dilate,erode);  
cv2.imwrite("out_morpho_gray.jpg",result)     

retval, result = cv2.threshold(result, 75, 255, cv2.THRESH_BINARY);
#result = cv2.erode(result, element)
cv2.imwrite("out_morpho.jpg",result)
close = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("out_morpho_close.jpg",close)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
open = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel)
cv2.imwrite("out_morpho_open.jpg",open)

#result = cv2.bitwise_not(result);

