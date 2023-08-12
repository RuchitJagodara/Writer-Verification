import cv2
import numpy as np


img=cv2.imread('A1.jpg')
height,width,_=img.shape

img_canny=cv2.Canny(img,350,130)
img_canny=cv2.dilate(img_canny,np.ones((3,3)),iterations=2)
def crop_fix(img,img_contours,img_original):
    contours,_=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i in contours:
        if cv2.contourArea(i)>10000:
            cv2.drawContours(img_contours,i,-100,[255,0,0],3)
            peri=cv2.arcLength(i,True)
            points=cv2.approxPolyDP(i,0.007*peri,True)
            point=[]
            for i in points:
                point.append(list(i[0]))
            point.sort(key= lambda x:x[0])
            width=(point[2][0]+point[3][0])/2-(point[1][0]+point[0][0])/2
            width=int(width)  
            if point[0][1]<point[1][1]:
                p1=point[0]
                p4=point[1]
                if point[2][1]<point[3][1]:
                    p2=point[2]
                    p3=point[3]
                else:
                    p2=point[3]
                    p3=point[4]
            else:
                p1=point[1]
                p4=point[0]
                if point[2][1]<point[1][1]:
                    p2=point[2]
                    p3=point[3]
                else:
                    p2=point[3]
                    p3=point[2]
            point.sort(key=lambda x:x[1])
            height=(point[3][1]+point[2][1])/2-(point[0][1]+point[1][1])/2
            height=int(height)
            ps2=np.float32([[0,0],[width,0],[width,height],[0,height]])
            point=np.float32([[p1[0]+10,p1[1]+10],[p2[0]-10,p2[1]+10],[p3[0]-10,p3[1]-10],[p4[0]+10,p4[1]-10]])   
            matrix=cv2.getPerspectiveTransform(point,ps2)
            img_final=cv2.warpPerspective(img_original,matrix,(width,height))
            return img_final
img_contours=np.ones_like(img)
final_image=crop_fix(img_canny,img_contours,img)