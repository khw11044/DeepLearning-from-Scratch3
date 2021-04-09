import numpy as np
import dezero
from dezero.core import Function, Variable, as_variable, as_array


# =============================================================================
# Basic functions: sin / cos / tanh / exp / log
# =============================================================================
class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)    # 1
        return gx

def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y 
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx 

def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self,x):
        y = np.tanh(x)
        return y 
    
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1-y*y)
        return gx 

def tanh(x):
    return Tanh()(x)

from dezero.core import add
from dezero.core import sub
from dezero.core import rsub
from dezero.core import mul
from dezero.core import div
from dezero.core import neg
from dezero.core import pow