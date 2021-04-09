import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:                            # 데이터가 None은 아닌데
            if not isinstance(data, np.ndarray):        # ndarray도 아니면      오류 발생
                raise TypeError("{}은(는) 지원하지 않습니다.".format(type(data)))
        
        self.data = data
        self.grad = None 
        self.creator = None 

    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() 
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)
                


def as_array(x):
    if np.isscalar(x):
        return np.array(x)      # 스칼라이면 array로 바꿔서 리턴
    return x                    # 스칼라 아니면(array이면) 그대로 리턴

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)