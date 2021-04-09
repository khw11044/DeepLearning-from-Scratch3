import weakref
import numpy as np
import contextlib


# =============================================================================
# Config
# =============================================================================
class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


# =============================================================================
# Variable / Function
# =============================================================================

class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):                            # 입력받는 데이터가 ndarray 구조가 아니면 오류 발생
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data        # 데이터 선언
        self.name = name
        self.grad = None        # 미분값 선언
        self.creator = None     # 이 데이터가 어디출신인지, 어느 공장에서 만들어졌는지 표기
        self.generation = 0     # 세대를 기록하는 변수
    
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func):    # 생성자 = 공장 = 함수
        self.creator = func 
        self.generation = func.generation + 1   # 세대를 기록한다(부모 세대 + 1)

    def cleargrad(self):
        self.grad = None                        # 미분값 초기화

    def backward(self, retain_grad=False):      # 추가되는곳 : 
        if self.grad is None:
            self.grad = np.ones_like(self.data)     # 미분값이 없으면 모두 1로 구성된 행렬

        funcs = []                      # <-- 바뀐부분
        seen_set = set()                # 집합은 중복을 막는다.

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)              # DezZero 함수 리스트를 세대순으로 정렬하는 역할 
                                            # 그결과 funcs.pop()은 자동으로 세대가 가장 큰 DeZero 함수 순으로 꺼낸다. 

        while funcs:
            f = funcs.pop()                         # 함수들을 하나씩 뽑는다.
            gys = [output().grad for output in f.outputs]     # 출력변수인 outputs에 담겨있는 미분값(.grad)들을 리스트에 담는다
            gxs = f.backward(*gys)                          # f의 역전파를 호출한다. *를 붙혀 리스트를 풀면서 넣어준다.(리스트 언팩)
            if not isinstance(gxs, tuple):                  # gxs가 튜플이 아니면 튜플로 변환한다.
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):                # gxs와 f.inputs의 각 원소는 서로 대응 관계
                if x.grad is None:
                    x.grad = gx                             # 역전파로 전파되는 미분값을 Variable의 인스턴스 변수 grad에 저장
                else:
                    x.grad = x.grad + gx    # x.grad가 None이 아니라 기존에 가지고 있는 값이 있다면 가지고 있는 값에 gx를 추가로 더한다.

                if x.creator is not None:
                    add_func(x.creator)      # <-- 바뀐부분, 수정전: funcs.append(x.creator) 출처가 있는 데이터를 add_funcs에 넣는다.
            
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None 



def as_variable(obj):
    if isinstance(obj, Variable):
        return obj 
    return Variable(obj)            # 인자로 받는 객체 인스턴스가 Variable이면 그대로 리턴, 아니면 Variable 씌어서 리턴


def as_array(x):
    if np.isscalar(x):
        return np.array(x)      # 스칼라이면 array로 바꿔서 리턴
    return x          

# =============================================================================
# 함수 
# =============================================================================

class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]           # 추가한 부분
        xs = [x.data for x in inputs]                       # 가변길이 인수를 다루기위해, 변수를 리스트에 담아 취급
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]       # 가변길이 입력이므로 가변길이 출력을 리스트로 담는다

        self.generation = max([x.generation for x in inputs])   # inputs의 generation중에 가장 큰것

        for output in outputs:
            output.set_creator(self)                     # 각각의 output들이 어디 출신 변수인지 정해짐, 자신이 창조자라고 원산지 표시를 함

        self.inputs = inputs
        #self.outputs = outputs
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]  # 리스트의 원소가 하나라면 첫번째 원소를 반환한다, 해당 변수를 직접 돌려준다

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


# =============================================================================
# 사칙연산 / 연산자 오버로드
# =============================================================================
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c

        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
