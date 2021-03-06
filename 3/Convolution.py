import numpy as np
from im2col import im2col
from col2im import col2im

class Convolution:
    def __init__(self, W, b, stride = 1, pad = 0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        FN, FC, FH, FW = self.W.shape

        out_H = int((H + 2*self.pad - FH) / self.pad + 1)
        out_W = int((W + 2 * self.pad - FW) / self.pad + 1)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FH, -1)
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_H, out_W, -1).transpose(0, 3, 1, 2)

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx