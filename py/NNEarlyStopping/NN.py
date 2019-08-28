from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import History, EarlyStopping

from keras.optimizers import SGD


import sys
from os import listdir
import random

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



def listFiles(_dir:str)->list:
 
  files=listdir(_dir)
  ariz=list(filter(lambda x: x.endswith('.ari'), files))
  
  random.seed(42)
  random.shuffle(ariz)

  cnt=0
  for i in ariz:
   ariz[cnt]=_dir+'/'+i
   cnt+=1
    

  return ariz

def read_img(_file:str)->list:
 file=_file
 
 byte_list:bytes=b''
 l=[]
 with open(_file,'rb') as f:
    
   byte_list=f.read()

 for i in byte_list:

   l.append(float(i/255.0))

 return l


def matrix_forNN(img_names:list)->(list,list):
  matrix=[]
  matrix_target=[]
  #matrix_targetRow=3*[0]

  for each_img in img_names:
      
     print(each_img,end=' ')
     matrix_targetRow=3*[0] 
     index=int(each_img[-5])
     matrix_targetRow[index]=1
     matrix_target.append(matrix_targetRow)
     matrix.append(read_img(each_img))
     print("class:",matrix_targetRow)
   
    
  return (matrix,matrix_target)

def create_and_learn_model(matrix,matrix_target,_epochs)->(Sequential,History):
  
   """
    EarlyStopping Callback в Keras наблюдает за метрикой качества обучения и
    прерывает обучение процесс обучения, если эта метрика начинает снижаться. 
    Метрика качества обучения задается в параметре monitor. Здесь мы использу
    ем долю правильных ответов на проверочном наборе данных ('val_acc').

    При обучении нейросети мы используем стохастический градиентный спуск или
    ана
    логичные методы, при которых качество решения на некоторых эпохах может снижа
    ться, но после этого снова возрастать. Параметр patience говорит о том, сколь
    ко эпох обучения может ухудшаться метрика качества, прежде чем обучение будет
    остановлено.
   """
   early_stopping_callback=EarlyStopping(monitor='val_acc',patience=2)
   model=Sequential()
  
   # 5 ти слойный перпецетрон
   model.add(Dense(30 ,input_dim=784,activation='sigmoid'))
   model.add(Dense(20 ,activation='sigmoid'))
   model.add(Dense(30 ,activation='sigmoid'))
   model.add(Dense(20 ,activation='sigmoid'))
   model.add(Dense(3,activation='sigmoid'))
   
   
   model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
   """
     Обучение
   """
   history:History=model.fit(matrix,matrix_target,batch_size=7,epochs=_epochs,validation_split=0.25,verbose=2,callbacks=[early_stopping_callback] )
 
   #get_model_summary(model) 
 
   return (model,history)

def save_model(_file:str,model:Sequential):
 
  model_json=model.to_json()
 
  with open(_file,'w') as f:
    f.write(model_json)

def plot_history(_file:str,history:History,_epochs:int):
  nRange=range(_epochs)
  plt.figure()
  plt.plot(history.history["loss"])
  plt.plot(history.history['acc'], 
         label='Доля верных ответов на обучающем наборе')
  plt.plot(history.history['val_acc'], 
         label='Доля верных ответов на проверочном наборе')
  plt.xlabel('Эпоха обучения')
  plt.ylabel('Доля верных ответов')
  plt.legend()
  plt.savefig(_file)

def save_weights(_file:str,model:Sequential):

  model.save_weights(_file)

if __name__=='__main__':
     
   dirRead:str=sys.argv[1]
   shuffledListFiles:list=listFiles(dirRead) 
   #print(shuffledListFiles)
   matrix,targets=matrix_forNN(shuffledListFiles)
   epochs=40
   model,history=create_and_learn_model(matrix,targets,epochs)

   save_weights('weights.h5',model)
   plot_history('./.graphik/graphik.png',history,epochs)

   
  
