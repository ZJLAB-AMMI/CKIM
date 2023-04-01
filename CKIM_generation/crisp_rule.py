import numpy as np
import os
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression as LR

path = "/home/ZP/CLEVR_v1.0/size_label/sample/"
files = os.listdir(path)

objs=[]

for i in range(len(files)):
  
  data = np.loadtxt(path+files[i]).tolist()
  
  for line in data:
    objs.append(line)

objs = np.array(objs)


DtoC = 1-objs[:,2]
Size = objs[:,3]*objs[:,4]


objs =np.vstack( [objs[:,0],DtoC,Size]).T
print(objs)

large_index = np.where(objs[:,0]<=47)
small_index = np.where(objs[:,0]>47)

large_objs = objs[large_index,:]
small_objs = objs[small_index,:]


small_data = np.vstack([small_objs[0][:,1],small_objs[0][:,2]]).T
small_label = np.zeros(small_data.shape[0])

large_data = np.vstack([large_objs[0][:,1],large_objs[0][:,2]]).T
large_label = np.ones(large_data.shape[0])

s_l_data = np.vstack([small_data,large_data])
s_l_label = np.hstack([small_label,large_label])


s_l_clf = LR(random_state=0).fit(s_l_data, s_l_label)

print(s_l_clf.coef_ , s_l_clf.intercept_) 
