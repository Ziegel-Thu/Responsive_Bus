import pandas as pd
import random 
import numpy as np
distance_matrix = pd.read_csv("wangjing-newdis-simplify.csv")
size = 73
expanded = pd.DataFrame(0,dtype= np.float64, index=range(size*3), columns=range(size*3))

for i in range(size):
    for j in range(size):
        dis = distance_matrix.iloc[i,j]
        dis = dis * random.uniform(1.5, 2)
        for k in range(3):
            for l in range (3):
                
                expanded.iloc[i+k*size,j+l*size] = dis

expanded.to_csv("wangjing-newdis-upflated.csv", index=False)
