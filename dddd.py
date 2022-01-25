import numpy as np
import cv2
a=np.zeros((500,500),dtype="uint8")
a[254][255]=255
a[255][255]=255
a[256][256]=255
a[257][257]=255
a[258][258]=255
a[259][259]=255
a[260][261]=255
a[256][256]=255
a[252][231]=255

a[351][351]=255
a[351][352]=255
a[352][352]=255
a[351][350]=255
a[349][350]=255

cv2.imshow("",a)
cv2.waitKey()
def imageReform(image):#2-Dimention
    filled_index=[]
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            if image[row][column]>=250 and ((row,column) not in filled_index):
                sizex=row
                sizey=column
                # print(row,column)
                for i in range(0,360,1):# 0 degree == 360 degree
                    length_x=int(5*np.cos(i*np.pi/180))
                    length_y=int(5*np.sin(i*np.pi/180))
                    if(sizex+length_x>image.shape[0]-1 or sizex+length_x<0):# 255인 곳이 0번인덱스이거나 마지막 인덱스인 경우..
                        # 단 아래 2*cos일 경우. x가 1일 때 -1이 되는데, 0번도 칠해줘야 하지만 이 경우는 무시하자..
                        length_x*=-1
                    if(sizey+length_y>image.shape[1]-1 or sizey+length_y<0):
                        length_y*=-1
                    if (length_x > 0 and length_y >0) or (length_x >0 and length_y==0) or (length_x ==0 and length_y > 0) :
                        for x in range(0,length_x+1):#+1인 0일 때 작동을 시키기 위해..추가함
                            for y in range(0,length_y+1):
                                # print(sizex+x,sizex+y)
                                image[sizex+x][sizey+y]=255
                                filled_index.append((sizex+x,sizey+y))
                    elif (length_x < 0 and length_y <0) or (length_x <0 and length_y==0) or (length_x ==0 and length_y <0) :
                        for x in range(0,length_x-1,-1):
                            for y in range(0,length_y-1,-1):
                                image[sizex+x][sizey+y]=255
                                filled_index.append((sizex+x,sizey+y))
                    elif (length_x<=0 and length_y>=0):
                        for x in range(0,length_x-1,-1):
                            for y in range(0,length_y+1,1):
                                image[sizex+x][sizey+y]=255
                                filled_index.append((sizex+x,sizey+y))
                    elif (length_x>=0 and length_y<=0):
                        for x in range(0,length_x+1,1):
                            for y in range(0,length_y-1,-1):
                                image[sizex+x][sizey+y]=255
                                filled_index.append((sizex+x,sizey+y))  
    return image

image=imageReform(a)
print(image.shape)
cv2.imshow("",image)
cv2.waitKey()