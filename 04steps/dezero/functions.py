import numpy as np
import dezero
from dezero import utils
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

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx


def exp(x):
    return Exp()(x)

class Log(Function):
    def forward(self, x):
        y = np.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx


def log(x):
    return Log()(x)

# =============================================================================
# Tensor operations: reshape / transpose / get_item / expand_dims / flatten
# =============================================================================

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape  # 클래스를 초기화할 때 변형 목표가 되는 형상을 shape 인수로 받는다.
    
    def forward(self, x):   # 순전파는 넘파이의 reshape 함수를 사용하여 형상을 변환한다.
        self.x_shape = x.shape  # 이때 입력 x의 shape을 기억해둔다.
        y = x.reshape(self.shape)
        return y 
    
    def backward(self, gy):     # 역전파에서 입력 shape(self.x_shape)으로 변환할 수 있다.
        return reshape(gy, self.x_shape)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)   # reshape 함수가 Variable 인스턴스를 반환함을 보장하기 위해 as_variable 함수를 사용하여 
    return Reshape(shape)(x)    # Variable 인스턴스를 보장

class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y 
    
    def backward(self, gy):
        gx = transpose(gy)
        return gx 

def transpose(x):
    return Transpose()(x)

class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices        # 슬라이스 연산을 수행하는 인수 slices를 받아 인스턴스 변수에 저장
    
    def forward(self, x):           
        y = x[self.slices]          # 단순히 slices 변수를 이용해 원소를 추출하고 있다
        return y 
    
    def backward(self, gy):         # 슬라이스 조작에 대응하는 역전파 계산이 DeZero 함수중에 없으므로
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)   # 새로 클래스를 만들어 사용해야 한다
        return f(gy)

def get_item(x, slices):
    return GetItem(slices)(x)

class GetItemGrad(Function):
    def __init__(self, slices, in_shape): # 초기화 메서드에서 슬라이스 연산 인수(slices)와 입력 데이터의 모양(in_shape)
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):      
        gx = np.zeros(self.in_shape)    # 입력용 기울기로서 원소가 모두 0인 다차원 배열 gx를 준비
        np.add.at(gx, self.slices, gy)  # gx의 원소 중 self.slices로 지정한 위치에 gy가 더해진다
        return gx 
    
    def backward(self, ggx):
        return get_item(ggx, self.slices)

# =============================================================================
# sum / sum_to / broadcast_to / average / matmul / linear
# =============================================================================

class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims) 
        return y 
    
    def backward(self, gy):
        gx = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y 
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx 

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)
    
class BroadcastTo(Function):
    def __init__(self,shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape 
        y = np.broadcast_to(x, self.shape)
        return y 
    
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx 
    
def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y 
    
    def backward(self, gy):
        x, W = self.inputs 
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW 

def matmul(x,W):
    return MatMul()(x,W)

class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t 
    
    y = t + b 
    t.data = None   # t의 데이터 삭제 
    return y 

# =============================================================================
# activation function: sigmoid / relu / softmax / log_softmax / leaky_relu
# =============================================================================
def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y

class Sigmoid(Function):
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx

def sigmoid(x):
    return Sigmoid()(x)

class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        y = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)

from dezero import Function
import numpy as np 

class ReLU(Function):
    def forward(self, x):
        y = np.maximum(x, 0.0)  # 1 x와 0.0중 큰것을 return 즉 x>0.0 이면 x, x<0.0이면 0.0 반환
        return y 
    
    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0       # 2 출력 쪽에서 전해지는 기울기를 '통과시킬지' 표시한 마스크를 준비
        gx = gy * mask          # 3 기울기를 곱해줌
        return gx 

def relu(x):
    return ReLU()(x)

# =============================================================================
# loss function: mean_squared_error / softmax_cross_entropy / sigmoid_cross_entropy / binary_cross_entropy
# =============================================================================
def mean_squared_error_simple(x0, x1):
    x0, x1 = as_variable(x0), as_variable(x1)
    diff = x0 - x1
    y = sum(diff ** 2) / len(diff)
    return y


class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]
    p = softmax(x)
    p = clip(p, 1e-15, 1.0)  # To avoid log(0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N
    return y


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        t_onehot = np.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)

def sofmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]

    p = softmax(x)  # 또는 softmax_simple(x)
    p = clip(p, 1e-15, 1.0) # log(0)을 방지하기 위해 p의 값을 1e-15 이상으로 한다. 
    log_p = log(p)  # log는 DeZero 함수 
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N
    return y

def sigmoid_cross_entropy(x, t):
    if x.ndim != t.ndim:
        t = t.reshape(*x.shape)
    x, t = as_variable(x), as_variable(t)
    N = len(x)
    p = sigmoid(x)
    p = clip(p, 1e-15, 1.0)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / N
    return y


def binary_cross_entropy(p, t):
    if p.ndim != t.ndim:
        t = t.reshape(*p.shape)
    N = len(t)
    p = clip(p, 1e-15, 0.999)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / N
    return y



# =============================================================================
# max / min / clip
# =============================================================================

class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        y = np.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


# =============================================================================
# accuracy / dropout / batch_norm / embed_id
# =============================================================================
def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)  # 텐서(ndarray)에 True/False 결과가 저장, 
    acc = result.mean()         # 이 텐서에 True 비율(평균)이 정답률
    return Variable(as_array(acc))

from dezero.core import add
from dezero.core import sub
from dezero.core import rsub
from dezero.core import mul
from dezero.core import div
from dezero.core import neg
from dezero.core import pow