
import numpy as np
from PIL import Image
from numpy.core.defchararray import count
import pandas as pd
import os
import numpy 
#필요한 것
#폴더 안에 모든 파일에 접근할 수 있어야 함
#파일 하나하나 자동으로 이름을 바꿀 수 있어야 함
#이제 각 파일에 다시 이 이름을 붙여 넣기 해주면 됨.
os.chdir("C:/Users/dlrms/OneDrive/Desktop/grayfolder")
img=os.listdir("C:/Users/dlrms/OneDrive/Desktop/grayfolder")
img_list=[]
def changenameandsize(path,cname):
    for index,name in enumerate(os.listdir(path)):
        imagepath=os.path.join(path,name)
        #print(imagepath)
        img=Image.open(imagepath).convert('L')
        img=img.resize((28,28))#여기까지는 아직.. array가 안 된거라, print해도 응답이 없음.
        #print(img)
        img_numpy=np.array(img,'uint8')
        #print("\ndasdasdas\n\n",img_numpy)
        img_list.append(img_numpy)
        #img_pandas=pd.DataFrame(img_numpy) # 여기서 이거 하면 안됨... 모든 list를 하나의 파일로 만들어 줘야함. 그래야만 해..!
        #img_pandas.to_csv("C:/Users/dlrms/OneDrive/Desktop")
        os.rename(path+name,cname+"kkkk"+str(index)+".jpg")#.jpg, .png, .. etc 아무거나 바꿔주면 다 바뀜.
        
changenameandsize("C:/Users/dlrms/OneDrive/Desktop/grayfolder/","number")
# img=img_list[0]
# img_form=Image.fromarray(img,'L')
# img_form.show()
# out=pd.read_csv("C:/Users/dlrms/OneDrive/Desktop/Mnist/mnist_test.csv")
# out.drop(axis=0,labels=["labels"],inplace=True)

#2차원 1차원으로 줄이기..28x28 => 784
def ChangeDemention(path,mode):
    empty_list=[]
    if(mode=="2D"):
        for index,image in enumerate(os.listdir(path)):
            imagepath=os.path.join(path,image)
            #print(imagepath)
            img=Image.open(imagepath).convert('L')
            #img=img.resize((10,715))
            #print(img)
            img_numpy=np.array(img,'uint8')#.tolist하니까 완전히 이상해짐.
            empty_list.append(img_numpy)
            #for row in range(1,len(image_array)):
        # import itertools as it
        # for sequance in range(0,len(empty_list)):
        #     empty_list[sequance]=list(it.chain.from_iterable(empty_list[sequance]))
        for sequance in range(0,len(empty_list)):
            for row in range(0,len(empty_list[sequance])):
                for colum in range(0,len(empty_list[sequance][row])):
                    if(empty_list[sequance][row][colum]<=115):
                        empty_list[sequance][row][colum]=255#255는 하얀색 0 은 검은색
                    #else: empty_list[sequance][row][colum]=255
        # img=empty_list[0]
        # img_form=Image.fromarray(img,'L')
        # img_form.show()
        for index,array in enumerate(empty_list):
            empty_list[index]=Image.fromarray(array).resize(())
            empty_list[index]=np.array(empty_list[index],'uint8')
        for sequance in range(0,len(empty_list)):#위에서 한 번 반전이 돼서 .. 
            for row in range(0,len(empty_list[sequance])):
                for colum in range(0,len(empty_list[sequance][row])):
                    if(empty_list[sequance][row][colum]<150):
                        empty_list[sequance][row][colum]=0#255는 하얀색 0 은 검은색
        #print(empty_list[0])#gray색상 판별용                   
        img=empty_list[0]
        img1=empty_list[1]
        img_form=Image.fromarray(img,'L')
        img_form2=Image.fromarray(img1,'L')
        img_form.show()
        img_form2.show()
        # for i,im in enumerate(empty_list):
        #     img_form=Image.fromarray(im[i],'L')
        #     img_form.show()
        # out=pd.read_csv("C:/Users/dlrms/OneDrive/Desktop/Mnist/mnist_test.csv")
        # out.drop(axis=0,labels=["labels"],inplace=True)
#pd.DataFrame(empty_list).to_csv("C:/Users/dlrms/OneDrive/Desktop/kkk.csv",)
ChangeDemention("C:/Users/dlrms/OneDrive/Desktop/grayfolder/","2D")
#ll=[[1,2],[2,3]]
#new=ll[0].extend(ll[1])
#print(new)
#print(len(img_list[0][0]),len(img_list[1][1]),len(img_list[1][2]))
#img_df=pd.DataFrame(img_list)
#print(img_df.info)
# 이제 각 파일의 이름과 접근에 성공했으니.. 이미지를 다뤄야함.
# 찍어 놓은 사진을 28 X 28로 변경..

#실험할 것, mnist csv의 행 하나를 복원해서 이미지로 띄우기.
#위에거 다 하면 ..마지막에는 csv로 저장해야함.
#csv의 구조는 각 행은 이미지의 행*열로 나열되어 있어야 하고 그 아래로 총 5개의 이미지 행이 존재 해야함
