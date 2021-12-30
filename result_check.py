import cv2
import numpy as np

refrence_image=cv2.imread(
    r"C:\Users\caoze\Downloads\A Grey Wolf Optimizer Based Automatic Clustering Algorithm for satellite image segmentation\unused material\Shanghai,_China.jpg"
)
#print(refrence_image[:,:,1])
print(np.shape(refrence_image))

b, g, r = cv2.split(refrence_image)
for i in range(10):
    for j in range(5):
        print(b[i,j])
r'''
test_image=cv2.imread(
    r"C:\Users\caoze\Downloads\A Grey Wolf Optimizer Based Automatic Clustering Algorithm for satellite image segmentation\czh_result,sta1\sat1_pso_otsu.png"
)
print(test_image[:,:,1])'''

