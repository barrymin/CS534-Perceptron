import numpy as np
       
#f1indx must be < f2indx
def combine_features(data,map,f1indx,f2indx):
    
    #extract features 
    r1 = sum([len(map[i]) for i in range(0,f1indx)])
    f1 = data[:,r1: len(map[f1indx])+r1]
    r2 = sum([len(map[i]) for i in range(0,f2indx)])
    f2 = data[:,r2: len(map[f2indx])+r2]
    # remove features from data
    data = np.delete(data,np.s_[r2: len(map[f2indx])+r2],1)
    data = np.delete(data,np.s_[r1: len(map[f1indx])+r1],1)
    #combine features
    features = (len(map[f2indx])*len(map[f1indx]))
    combined = np.zeros((0,features))
    for i in range(0,len(data)):
        x = np.outer(f1[i],f2[i])
        x = x.reshape(1,features)
        combined = np.append(combined,x,axis=0)
        
    #add combined features to data
    return np.hstack((data,combined))
    
