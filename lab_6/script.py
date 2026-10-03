from numba import cuda
from matplotlib import pyplot
import numpy as np
import time

src = pyplot.imread("input/big-cat.jpg")
src2 = pyplot.imread("input/resized_big-frog.jpg")
flatSrc = src.reshape(-1, 3)
height = src.shape[0]
width = src.shape[1]
pixelCount = height * width

devSrc2 = cuda.to_device(src2)
devSrc = cuda.to_device(src)

devDstBlend = cuda.to_device(np.zeros((height, width, 3), np.uint8))
devDstBin = cuda.to_device(np.zeros((height, width, 3), np.uint8))
devDstBri = cuda.to_device(np.zeros((height, width, 3), np.uint8))

blockDim = 16
blockSize = (blockDim, blockDim)
gridSize = ((width + blockDim - 1) // blockDim, (height + blockDim -1) // blockDim)
T = 120
V = 30
C = 0.5

@cuda.jit
def binarization(src, dst):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if y > src.shape[0] or x > src.shape[1]:
        return
    g = np.uint8((src[y, x, 0] + src[y, x, 1] + src[y, x, 2]) / 3)
    if (g > T):
        dst[y, x, 0] = dst[y, x, 1] = dst[y, x, 2] = 0
    else:
        dst[y, x, 0] = dst[y, x, 1] = dst[y, x, 2] = 255

@cuda.jit
def brightness(src, dst):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if y > src.shape[0] or x > src.shape[1]:
        return
    dst[y, x, 0] = max(min((src[y, x, 0] + V), 255), 0)
    dst[y, x, 1] = max(min((src[y, x, 0] + V), 255), 0)
    dst[y, x, 2] = max(min((src[y, x, 0] + V), 255), 0)

@cuda.jit
def blend(src1, src2, dst):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if y > src1.shape[0] or x > src1.shape[1] or src1.shape[0] != src2.shape[0] or src1.shape[1] != src2.shape[1]:
        return
    dst[y, x, 0] = C * src1[y, x, 0] + (1 - C) * src2[y, x, 0]
    dst[y, x, 1] = C * src1[y, x, 1] + (1 - C) * src2[y, x, 1]
    dst[y, x, 2] = C * src1[y, x, 2] + (1 - C) * src2[y, x, 2]

blend[gridSize, blockSize](devSrc, devSrc2, devDstBlend)
binarization[gridSize, blockSize](devSrc, devDstBin)
brightness[gridSize, blockSize](devSrc,  devDstBri)

hostDstBlend = devDstBlend.copy_to_host()
hostDstBin = devDstBin.copy_to_host()
hostDstBri = devDstBri.copy_to_host()
pyplot.imsave("output/resultBlend.jpg", hostDstBlend)
pyplot.imsave("output/resultBin.jpg", hostDstBin)
pyplot.imsave("output/resultBri.jpg", hostDstBri)