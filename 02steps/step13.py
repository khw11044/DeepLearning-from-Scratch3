import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):                            # 입력받는 데이터가 ndarray 구조가 아니면 오류 발생
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data        # 데이터 선언
        self.grad = None        # 미분값 선언
        self.creator = None     # 이 데이터가 어디출신인지, 어느 공장에서 만들어졌는지 표기

    def set_creator(self, func):    # 생성자 = 공장 = 함수
        self.creator = func 

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)     # 미분값이 없으면 모두 1로 구성된 행렬

        funcs = [self.creator]                      # 함수들을 담는 리스트 
        while funcs:
            f = funcs.pop()                         # 함수들을 하나씩 뽑는다.
            gys = [output.grad for output in f.outputs]     # 출력변수인 outputs에 담겨있는 미분값(.grad)들을 리스트에 담는다
            gxs = f.backward(*gys)                          # f의 역전파를 호출한다. *를 붙혀 리스트를 풀면서 넣어준다.(리스트 언팩)
            if not isinstance(gxs, tuple):                  # gxs가 튜플이 아니면 튜플로 변환한다.
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):                # gxs와 f.inputs의 각 원소는 서로 대응 관계
                x.grad = gx                                 # 역전파로 전파되는 미분값을 Variable의 인스턴스 변수 grad에 저장

                if x.creator is not None:
                    funcs.append(x.creator)



def as_array(x):
    if np.isscalar(x):
        return np.array(x)                                  # 입력이 스칼라인 경우 ndarray 인스턴스로 변화해줌
    return x


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]                       # 가변길이 인수를 다루기위해, 변수를 리스트에 담아 취급
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]       # 가변길이 입력이므로 가변길이 출력을 리스트로 담는다

        for output in outputs:
            output.set_creator(self)                        # 각각의 output들이 어디 출신 변수인지 정해짐, 자신이 창조자라고 원산지 표시를 함
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]  # 리스트의 원소가 하나라면 첫번째 원소를 반환한다, 해당 변수를 직접 돌려준다

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x):
    f = Square()
    return f(x)


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


# x = Variable(np.array(2.0))
# y = Variable(np.array(3.0))

# z = add(square(x), square(y))
# z.backward()
# print(z.data)
# print(x.grad)
# print(y.grad)