import numba
from numba import cuda
from matplotlib import pyplot
import time
import numpy as np

src = pyplot.imread("input/small-cat.jpg")
height = src.shape[0]
width = src.shape[1]
pixelCount = height * width

blockDim = 16
blockSize = (blockDim, blockDim)
gridSize = ((width + blockDim - 1) // blockDim, (height + blockDim -1) // blockDim)
rangeNumber = 256

devSrc= cuda.to_device(src)
devGray = cuda.device_array((height, width), int)
devHistOriginal = cuda.to_device(np.zeros((rangeNumber), int))
devHistEqualized = cuda.to_device(np.zeros((rangeNumber), int))
devDst = cuda.device_array((height, width), int)

@cuda.jit
def grayscale(src, dst):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if y >= src.shape[0] or x >= src.shape[1]:
        return
    g = numba.uint8((src[y, x, 0] + src[y, x, 1] + src[y, x, 2]) / 3)
    dst[y, x] = g

@cuda.jit
def histogram(src, dst):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    tid = cuda.threadIdx.y * 16 + cuda.threadIdx.x
    if y > src.shape[0] or x > src.shape[1]:
        return
    sharedHist = cuda.shared.array(256, np.int32)
    sharedHist[tid] = 0

    cuda.syncthreads()
    cuda.atomic.add(sharedHist, src[y, x], 1)
    cuda.syncthreads()

    cuda.atomic.add(dst, tid, sharedHist[tid])

@cuda.jit
def equalize(src, hist, dst):
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if y > src.shape[0] or x > src.shape[1]:
        return

    c = 0
    for i in range(src[y, x]):
        c += hist[i] / pixelCount

    dst[y, x] = c * 255

grayscale[gridSize, blockSize](devSrc, devGray)
histogram[gridSize, blockSize](devGray, devHistOriginal)
equalize[gridSize, blockSize](devGray, devHistOriginal, devDst)
histogram[gridSize, blockSize](devDst, devHistEqualized)

histOriginal = devHistOriginal.copy_to_host()
histEqualized = devHistEqualized.copy_to_host()
imageGray = devGray.copy_to_host()
imageEqualized = devDst.copy_to_host()

pyplot.figure()
pyplot.bar(range(256), histEqualized)
pyplot.savefig("output/histoEqualized.jpg")

pyplot.figure()
pyplot.bar(range(256), histOriginal)
pyplot.savefig("output/histOriginal.jpg")

pyplot.imsave("output/result.jpg", imageEqualized, cmap='gray')
pyplot.imsave("output/original.jpg", imageGray, cmap='gray')