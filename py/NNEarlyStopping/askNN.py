from keras.models import Sequential
from keras.layers import Dense

import numpy as np

import h5py



def read_img(_file:str)->np.ndarray:
 file=_file
 
 byte_list:bytes=b''
 l=[]
 with open(_file,'rb') as f:
    
   byte_list=f.read()

 for i in byte_list:

   l.append(float(i/255.0))

 return np.array([l])


def ask(ask_matr:list):

   model=Sequential()
  
   # 5 ти слойный перпецетрон
   model.add(Dense(30 ,input_dim=784,activation='sigmoid'))
   model.add(Dense(20 ,activation='sigmoid'))
   model.add(Dense(30 ,activation='sigmoid'))
   model.add(Dense(20 ,activation='sigmoid'))
   model.add(Dense(3,activation='sigmoid'))

   model.load_weights('weights.h5') 

   answ=model.predict(ask_matr)
   print(answ,type(answ)) 

if __name__=='__main__':
  import sys
  ask(read_img(sys.argv[1]))
