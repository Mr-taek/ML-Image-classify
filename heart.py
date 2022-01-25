# def standardization(data):#age,cp,trestbps,chol,thalach,oldpeak Stadardization
#     label=data.columns#['age','trestbps','chol','thalach','oldpeak','slope']
#     for i in range(len(label)):
#         dvias=np.sqrt(np.sum((data[label[i]]-np.average(data[label[i]]))**2)/data[label[i]].shape[0]-10*np.exp(-7))
#         data[label[i]]=(data[label[i]]-np.average(data[label[i]]))/dvias
#     return data
# data=pd.read_csv('C:/Users/dlrms/OneDrive/Desktop/joljak/archive/heart.csv')
# temp=data.drop(labels=["cp"],axis=1)
# data=pd.concat([temp,pd.get_dummies(data['cp'])],axis=1)#pd.get_dummies(data['thal'],pd.get_dummies(data['ca'])
# data.rename({0:'cp0',1:'cp1',2:'cp2',3:'cp3'},axis=1,inplace=True)
# data=standardization(data)

#print(data[:10][['age','trestbps','chol','thalach','oldpeak','slope']])
# data shuffle
# for i in range(data.shape[0]):
#     ran=np.random.randint(0,data.shape[0])
#     temp=data.iloc[ran]
#     data.iloc[ran]=data.iloc[i]
#     data.iloc[i]=temp
# x=data.drop(labels=['target'],axis=1)
# y=data['target']
# x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7)#random 기능이 있어서 앞으로 위에 저거 할 필요는 없을듯
# y_test=pd.get_dummies(y_test)
# x_test=np.array(x_test)
# y_test=np.array(y_test)