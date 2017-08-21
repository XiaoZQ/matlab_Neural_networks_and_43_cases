from Network import BPNuNeuralNetwork
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from activation_function import sigmoid
# import os
# print(os.getcwd())
data1=sio.loadmat('data1.mat')
data2=sio.loadmat('data2.mat')
data3=sio.loadmat('data3.mat')
data4=sio.loadmat('data4.mat')

c1=data1['c1']
c2=data2['c2']
c3=data3['c3']
c4=data4['c4']

data=np.concatenate((c1,c2,c3,c4),axis=0)

rows,cols=data.shape
input_data=data[:,1:]
output_data1=data[:,0]

output_data=np.zeros((rows,4))

for i in range(1,5):
    output_data[output_data1==i,i-1]=1

np.random.seed(1000)
k=np.random.choice(range(input_data.shape[0]),2000,replace=False)

input_train=input_data[k[:1500],:]
output_train=output_data[k[:1500],:]
input_test=input_data[k[1500:],:]
output_test=output_data[k[1500:],:]


def mapmaxmin(data):
    min_data=data.min(axis=0)
    max_data=data.max(axis=0)
    return (data-min_data)/(max_data-min_data),np.array([max_data,min_data])


inputn,inputps=mapmaxmin(input_train)
inputn_test=(input_test-inputps[1,:])/(inputps[0,:]-inputps[1,:])
neww=BPNuNeuralNetwork(inputn,output_train,[25,24],activation_fun=sigmoid)
neww.train(epochs=100)
# print(output_test,neww.w)
# print(inputn_test)
fore=neww.predict(inputn_test)

output_fore=fore.max(axis=1)
output_fore=np.reshape(output_fore,(len(output_fore),1))

right_data=np.where(output_test-np.equal(fore,output_fore)[1]<0,False,output_test-np.equal(fore,output_fore)[1])
print((len(right_data)-right_data.sum(axis=0))/len(right_data))