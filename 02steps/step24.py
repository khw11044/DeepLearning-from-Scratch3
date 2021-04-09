if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable

def sphere(x,y):
    z = x ** 2 + y ** 2
    return z 

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = sphere(x,y)
z.backward
print(x.grad, y.grad)