import numpy as np
from im2col import im2col
from col2im import col2im

class Pooling:
    def __init__(self, pool_h, pool_w, stride = 1, pad = 0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        FN, FC, FH, FW = self.W.shape

        out_H = int((H + 2*self.pad - FH) / self.pad + 1)
        out_W = int((W + 2 * self.pad - FW) / self.pad + 1)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FH, -1)
        out = np.max(col, axis = 1)

        out = out.reshape(N, out_H, out_W, C).transpose(0, 3, 1, 2)

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx