import numpy as np

class ReluLayer:
    def __init__(self):
        self.Mask = None

    def forward(self, x):
        self.Mask = x <= 0
        out = x.copy()
        out[self.Mask] = 0
        return out

    def backward(self,dout):
        dout[self.Mask] = 0
        return dout