from numba import cuda
from matplotlib import pyplot
import numpy as np
import time

src = pyplot.imread("input/big-cat.jpg")
flatSrc = src.reshape(-1, 3)
height = src.shape[0]
width = src.shape[1]

blockDim = 16
blockSize = (blockDim, blockDim)
gridSize = ((width + blockDim - 1) // blockDim, (height + blockDim -1) // blockDim)

devSrcRGB2HSV = cuda.to_device(src)
devDstRGB2HSV = cuda.device_array((3, height, width), np.float32)

devDstHSV2RGB = cuda.device_array((height, width, 3), np.uint8)

@cuda.jit
def RGB2HSV(src, dst):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x >= src.shape[1] or y >= src.shape[0]:
        return

    r = src[y, x, 0] / 255
    g = src[y, x, 1] / 255
    b = src[y, x, 2] / 255
    maxV = max(r, g, b)
    minV = min(r, g, b)
    delta = maxV - minV
     
    if (maxV == 0):
        dst[1, y, x] = maxV
    else:
        dst[1, y, x] = delta / maxV

    dst[2, y, x] = maxV
    if delta == 0:
        dst[0, y, x] = 0
    elif maxV == r:
        dst[0, y, x] = 60 * (((g - b) / delta) % 6)
    elif maxV == g:
        dst[0, y, x] = 60 * ((b - r) / delta + 2)
    else:
        dst[0, y, x] = 60 * ((r - g) / delta + 4)

@cuda.jit
def HSV2RGB(src, dst):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x >= src.shape[2] or y >= src.shape[1]:
        return
    h = src[0, y, x]
    s = src[1, y, x]
    v = src[2, y, x]

    d = h / 60
    f = d - int(d)
    l = v * (1 - s)
    m = v * (1 - f * s)
    n = v * (1 - (1 - f) * s)

    if (0 <= h < 60):
        dst[y, x, 0] = round(v * 255)
        dst[y, x, 1] = round(n * 255)
        dst[y, x, 2] = round(l * 255)
    elif (60 <= h < 120):
        dst[y, x, 0] = round(m * 255)
        dst[y, x, 1] = round(v * 255)
        dst[y, x, 2] = round(l * 255)
    elif (120 <= h < 180):
        dst[y, x, 0] = round(l * 255)
        dst[y, x, 1] = round(v * 255)
        dst[y, x, 2] = round(n * 255)
    elif (180 <= h < 240):
        dst[y, x, 0] = round(l * 255)
        dst[y, x, 1] = round(m * 255)
        dst[y, x, 2] = round(v * 255)
    elif (240 <= h < 300):
        dst[y, x, 0] = round(n * 255)
        dst[y, x, 1] = round(l * 255)
        dst[y, x, 2] = round(v * 255)
    else:
        dst[y, x, 0] = round(v * 255)
        dst[y, x, 1] = round(l * 255)
        dst[y, x, 2] = round(m * 255)

RGB2HSV[gridSize, blockSize](devSrcRGB2HSV, devDstRGB2HSV)
HSV2RGB[gridSize, blockSize](devDstRGB2HSV, devDstHSV2RGB)
rst = devDstHSV2RGB.copy_to_host()
pyplot.imsave("output/result.jpg", rst)