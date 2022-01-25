data=pd.read_csv('C:/Users/dlrms/OneDrive/Desktop/joljak/archive/heart.csv')
for i in range(data.shape[0]):
    index=np.random.randint(0,data.shape[0])
    tempa=data.iloc[i]
    data.iloc[i]=data.iloc[index]
    data.iloc[index]=tempa