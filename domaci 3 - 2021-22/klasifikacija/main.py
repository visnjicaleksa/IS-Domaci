import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#Problem statement and read data
pd.set_option("display.max_columns",15)
pd.set_option("display.width",None)
data=pd.read_csv("datasets/car_state.csv")
print(data)

#Data analysis
print(data.info())
#feature statistic
print(data.describe())
print(data.describe(include=[object]))
#,["0","1","2","3","4","5 or more"], ["small","medium", "big"], ["unacceptable", "acceptable", "good", "very good"]
#useful features: buying_price, maintainance, doors, seats, trunk_size, safety
#not-useful: customer_id
data_train=data.loc[:, ["buying_price", "maintenance", "doors", "seats", "trunk_size", "safety"]] #DataFrame
labels=data.loc[:,"status"] #series
oe=OrdinalEncoder(categories=[["low","medium","high","very high"]])
le=LabelEncoder()
#bying price conversion
bprice=oe.fit_transform(data_train.buying_price.to_numpy().reshape(-1,1))
bprice=bprice.astype(int)
data_train.drop(columns=["buying_price"], inplace=True)
data_train=data_train.join(pd.DataFrame(data=bprice,columns=["buying_price"]))
#maintenance conversion
bprice=oe.fit_transform(data_train.maintenance.to_numpy().reshape(-1,1))
bprice=bprice.astype(int)
data_train.drop(columns=["maintenance"], inplace=True)
data_train=data_train.join(pd.DataFrame(data=bprice,columns=["maintenance"]))
#doors conversion
oe.categories=[["0","1","2","3","4","5 or more"]]
bprice=oe.fit_transform(data_train.doors.to_numpy().reshape(-1,1))
bprice=bprice.astype(int)
data_train.drop(columns=["doors"], inplace=True)
data_train=data_train.join(pd.DataFrame(data=bprice,columns=["doors"]))
#seats conversion
bprice=oe.fit_transform(data_train.seats.to_numpy().reshape(-1,1))
bprice=bprice.astype(int)
data_train.drop(columns=["seats"], inplace=True)
data_train=data_train.join(pd.DataFrame(data=bprice,columns=["seats"]))
#trunk_size conversion
oe.categories=[["small","medium", "big"]]
bprice=oe.fit_transform(data_train.trunk_size.to_numpy().reshape(-1,1))
bprice=bprice.astype(int)
data_train.drop(columns=["trunk_size"], inplace=True)
data_train=data_train.join(pd.DataFrame(data=bprice,columns=["trunk_size"]))
#safety conversion
oe.categories=[["low","medium","high","very high"]]
bprice=oe.fit_transform(data_train.safety.to_numpy().reshape(-1,1))
bprice=bprice.astype(int)
data_train.drop(columns=["safety"], inplace=True)
data_train=data_train.join(pd.DataFrame(data=bprice,columns=["safety"]))

print(data_train)

oe.categories=[["unacceptable", "acceptable", "good", "very good"]]
bprice=oe.fit_transform(labels.to_numpy().reshape(-1,1))
bprice=bprice.astype(int)
print(labels)
labels=pd.Series(bprice.flatten())
print(labels)

data_train.buying_price=data_train.buying_price/3
data_train.maintenance=data_train.maintenance/3
data_train.doors=(data_train.doors-2)/3
data_train.seats=(data_train.seats-2)/3
data_train.trunk_size=data_train.trunk_size/2
data_train.safety=data_train.safety/2
labels=labels/3

plt.figure("buying price")
plt.scatter(data_train.buying_price, labels, s=23, color="red",label="cars")
plt.xlabel('buying price', fontsize=13)
plt.ylabel('status', fontsize=13)
plt.title('buying price')
plt.legend()
plt.show()

plt.figure("maintenance")
plt.scatter(data_train.maintenance, labels, s=23, color="blue",marker="x",label="cars")
plt.xlabel('Maintenance', fontsize=13)
plt.ylabel('status', fontsize=13)
plt.title('maintenance')
plt.legend()
plt.show()

plt.figure("doors")
plt.scatter(data_train.doors, labels, s=23, color="green",marker="v",label="cars")
plt.xlabel('Doors', fontsize=13)
plt.ylabel('status', fontsize=13)
plt.title('Doors')
plt.legend()
plt.show()

plt.figure("seats")
plt.scatter(data_train.seats, labels, s=23, color="black",marker="d",label="cars")
plt.xlabel('Seats', fontsize=13)
plt.ylabel('status', fontsize=13)
plt.title('Seats')
plt.legend()
plt.show()

plt.figure("trunk_size")
plt.scatter(data_train.trunk_size, labels, s=23, color="purple",marker=".",label="cars")
plt.xlabel('Trunk size', fontsize=13)
plt.ylabel('status', fontsize=13)
plt.title('Trunk size')
plt.legend()
plt.show()

plt.figure("safety")
plt.scatter(data_train.safety, labels, s=18, color="brown",marker="D",label="cars")
plt.xlabel('Safety', fontsize=13)
plt.ylabel('status', fontsize=13)
plt.title('Safety')
plt.legend()
plt.show()

xtrain, xtest, ytrain, ytest=train_test_split(data_train, labels, train_size=0.7, random_state=123, shuffle=False)
class KNajblizihSuseda:
    def __init__(self):
        self.x = None
        self.y = None
        self.k = None
        self.width = None

    def setData(self, xdata, ydata):
        self.x=xdata.values
        self.y=ydata.to_numpy()
        self.k=int(np.sqrt(len(self.x)))
        if(self.k%2==0):
            self.k+=1
        self.width=len(self.x[0])

    def pronalazakNajblizeg(self, data):
        klasa=[0,0,0,0]
        klasa=np.array(klasa)
        dist=[]
        for i in range(0,len(self.x)):
            d=0
            for j in range(0,self.width):
                t=(data[j]-self.x[i][j])**2
                d+=t
            d=np.sqrt(d)
            dist.append(d)
        dist=np.array(dist)
        for i in range(0,self.k):
            m=np.argmin(dist)
            ka=np.ceil(self.y[m]*3)
            ka=int(ka)
            klasa[ka]+=1
            np.delete(dist, m)
        return np.argmax(klasa)

    def knn(self, xtest):
        x=xtest.values
        y=[]
        for i in range(0,len(x)):
            ka=self.pronalazakNajblizeg(x[i])
            ka=ka/3
            y.append(ka)
        return pd.Series(np.array(y))

def funkcija_greske(ytest, yrez):
    ytest2=ytest.to_numpy()
    yrez2=yrez.to_numpy()
    rez=0
    for i in range(0, ytest.size):
        rez+=(ytest2[i]-yrez2[i])**2
    rez/=ytest.size
    return rez

def preciznost(ytest, yrez):
    ytest2 = ytest.to_numpy()
    yrez2 = yrez.to_numpy()
    rez = 0
    for i in range(0, ytest.size):
        if(ytest2[i]==yrez2[i]):
            rez+=1
    rez /= ytest.size
    return rez

alg=KNajblizihSuseda()
alg.setData(xtrain, ytrain)
yrez=alg.knn(xtest)
print("Funkcija greske za moj algoritam je: ",funkcija_greske(ytest, yrez))
print("Preciznost mog algoritma je: ",preciznost(ytest, yrez))
print("\n")
xtest2=xtest.values
nn=int(np.sqrt(len(xtest2)))
print(nn)
kncl=KNeighborsClassifier(n_neighbors=3)




#xtrain.buying_price=int(np.ceil(xtrain.buying_price*3))
#xtrain.maintenance=int(np.ceil(xtrain.maintenance*3))
#xtrain.doors=int(np.ceil(xtrain.doors*3))
#xtrain.seats=int(np.ceil(xtrain.seats*3))
#xtrain.trunk_size=int(np.ceil(xtrain.trunk_size*2))
#xtrain.safety=int(np.ceil(xtrain.safety*2))
#ytrain=int(np.ceil(ytrain*3))
ytrain=ytrain*3
for i in ytrain:
    i=int(np.ceil(i))
ytest = ytest * 3
for i in ytest:
    i = int(np.ceil(i))
print(xtrain)
print(ytrain)
kncl.fit(xtrain, ytrain)
ypred=kncl.predict(xtest)
print("Funkcija greske za ugradjeni algoritam je: ", mean_squared_error(ytest, ypred))
print("Preciznost ugradjenog algoritma je: ", kncl.score(xtest, ytest))


