files = [s for s in all_files if 'txt' in s]

objs=[]

for i in range(len(files)):
  
  data = np.loadtxt(path+files[i]).tolist()
  
  for line in data:
    objs.append(line)

objs = np.array(objs)

S_index = np.where(objs[:,0]%3==1)

DtoC = 1-objs[:,2]
Size = objs[:,3]*objs[:,4]*1000

objs =np.vstack( [objs[:,0],DtoC,Size]).T

large_index = np.where(objs[:,0]<=47)
small_index = np.where(objs[:,0]>47)

large_objs = objs[large_index,:]
small_objs = objs[small_index,:]


labels = objs[:,0]
labels[large_index]=0
labels[small_index]=1


small_mean = np.mean(small_objs[0][:,1:] , axis=0)
small_cov = np.cov(small_objs[0][:,1:] , rowvar=0)


large_mean = np.mean(large_objs[0][:,1:] , axis=0)
large_cov = np.cov(large_objs[0][:,1:] , rowvar=0)


with open('fuzzy_test.txt','w') as test:
  test.write('samll_mean = '+str(small_mean)+'\n')
  test.write('samll_cov = '+str(small_cov)+'\n')
  test.write('large_mean = '+str(large_mean)+'\n')
  test.write('large_cov = '+str(large_cov)+'\n')
  
  
  
  
  