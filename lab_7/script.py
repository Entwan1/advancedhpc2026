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

devSrc = cuda.to_device(src)
devGray = cuda.device_array((height, width), np.uint8)
devDstScale = cuda.device_array((height, width), np.uint8)
devMin = cuda.device_array(gridSize, np.uint8)
devMax = cuda.device_array(gridSize, np.uint8)
devDst = cuda.device_array((height, width), np.uint8)

@cuda.jit
def grayscale(src, dst):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if y > src.shape[0] or x > src.shape[1]:
        return
    g = np.uint8((src[y, x, 0] + src[y, x, 1] + src[y, x, 2]) / 3)
    dst[y, x] = g

@cuda.jit
def reduce(src, minBlock, maxBlock):
    shared_min = cuda.shared.array((blockDim, blockDim), np.uint8)
    shared_max = cuda.shared.array((blockDim, blockDim), np.uint8)
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    xInBlock = cuda.threadIdx.x
    yInBlock = cuda.threadIdx.y
    if x > src.shape[1] or y > src.shape[0]:
        return
    shared_min[yInBlock][xInBlock] = src[y, x]
    shared_max[yInBlock][xInBlock] = src[y, x]
    cuda.syncthreads()

    startX = blockDim // 2
    while startX > 0:
        if xInBlock < startX:
            shared_min[yInBlock][xInBlock] = min(shared_min[yInBlock][xInBlock], shared_min[yInBlock][xInBlock + startX])
            shared_max[yInBlock][xInBlock] = max(shared_max[yInBlock][xInBlock], shared_max[yInBlock][xInBlock + startX])
        cuda.syncthreads()
        startX = startX // 2

    startY = blockDim // 2
    while startY > 0:
        if xInBlock == 0 and yInBlock < startY:
            shared_min[yInBlock][0] = min(shared_min[yInBlock][0], shared_min[yInBlock + startY][0])
            shared_max[yInBlock][0] = max(shared_max[yInBlock][0], shared_max[yInBlock + startY][0])
        cuda.syncthreads()
        startY = startY // 2

    minBlock[cuda.blockIdx.x, cuda.blockIdx.y] = shared_min[0, 0]
    maxBlock[cuda.blockIdx.x, cuda.blockIdx.y] = shared_max[0, 0]

@cuda.jit
def stretch(gray, dst, gMin, gMax):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if y >= gray.shape[0] or x >= gray.shape[1]:
        return
    dst[y, x] = (int(gray[y, x]) - gMin) * 255 // (gMax - gMin)

grayscale[gridSize, blockSize](devSrc, devGray)

reduce[gridSize, blockSize](devGray, devMin, devMax)
gMin = int(devMin.copy_to_host().min())
gMax = int(devMax.copy_to_host().max())

stretch[gridSize, blockSize](devGray, devDst, gMin, gMax)

hostDst = devDst.copy_to_host()
pyplot.imsave("output/result.jpg", hostDst, cmap='gray')
