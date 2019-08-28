import h5view
import numpy as np

def extract_matrices(_file:str)->{}:
    matrices={}
    with h5view.open(_file) as f:
        
        list_layer_names=f.get('layer_names')
               
        print(list_layer_names)
        
        for i in range(len(list_layer_names)):
             dense=[]#матрица
             bias=[]             

             layer_name=list_layer_names[i].decode('cp1251')
             print(layer_name)
      
            
             for row in f.get(layer_name+'/'+layer_name+'/kernel').item():
               dense.append(row)
             try:
               for row in f.get(layer_name+'/'+layer_name+'/bias').item():
                 bias.append(row)
                 print(row)
             except Exception:
                pass  

             dense=np.array(dense)
             dense=dense.T

             matrices[layer_name]=dense.tolist()

             matrices[layer_name+'_bias']=bias
            
         
        return matrices

def change_to_C_matrices(matrices:str)->str:
 list_C_matr:list=[' ']*len(matrices)
 cnt=0
 for i in matrices:
   if i=='[':
      list_C_matr[cnt]='{'

   elif i==']':
       list_C_matr[cnt]='}'

   elif i.isdigit():
       list_C_matr[cnt]=i

   elif i=='-' or i=='.' or i==',' or 'e':
       list_C_matr[cnt]=i

   cnt+=1
 return ''.join(list_C_matr)

def write_C_matrices(_file:str,matrices:dict):
  
   str_to_file=''

   for i in matrices.items():
      str_to_file+=i[0]+'='
      str_to_file+=change_to_C_matrices(repr(i[1]))+';\n'
      
   with open(_file,'w') as f:
     f.write(str_to_file) 

import sys    
matrices=extract_matrices(sys.argv[1])
#print(matrices)
write_C_matrices('C_weights.txt',matrices)
        


   
