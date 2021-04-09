import os
import weakref
import numpy as np
import dezero.functions as F
from dezero.core import Parameter


class Layer:
    def __init__(self):
        self._params = set()
    
    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):   # 1. Layer도 추가 
            self._params.add(name)
        super().__setattr__(name, value)
    
    def params(self):
        for name in self._params:       # _params에서 name을 꺼내
            obj = self.__dict__[name]   # 그 name에 해당하는 객체를 obj로 꺼냄

            if isinstance(obj, Layer):  # 2. Layer에서 매개변수 꺼내기 
                yield from obj.params()
            else:
                yield obj               
    
    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, inputs):
        raise NotImplementedError()

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()



class Linear(Layer):
    # def __init__(self, in_size, out_size, nobias=False, dtype=np.float32):  # 입력크기, 출력크기, 편향사용여부
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size      # <- 추가 
        self.out_size = out_size    # <- 추가
        self.dtype = dtype          # <- 추가

        # I, O = in_size, out_size
        # W_data = np.random.randn(I,O).astype(dtype) * np.sqrt(1/I)  # (0.01 * np.random.randn()대신)
        # self.W = Parameter(W_data, name='W') # 아래 코드로 바뀜 
        self.W = Parameter(None, name='W')
        if self.in_size is not None:    # in_size가 지정되어 있지 않다면 나중으로 연기
            self._init_W()
            
        if nobias:
            self.b = None 
        else:
            # self.b = Parameter(np.zeros(0, dtype=dtype), name='b')
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')
        # self.W와 self.b가 Parameter 형태로, 두 Parameter 인스턴스 변수의 이름이 self._params에 추가된다.
    
    # 추가
    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data  
    
    def forward(self, x):               # 선형 변환 구현
        # 데이터를 흘려보내는 시점에 개충치 초기화 
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()
            
        y = F.linear(x, self.W, self.b)
        return y 