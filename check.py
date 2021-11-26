from numpy.core.fromnumeric import argmax
import numpy as np

Q  = {'1':3,'2':4}

def doit(Q):
    def change():
        print(Q)
    return change
hi = doit(Q)
hi()
Q['1']=10
hi()
print(Q)