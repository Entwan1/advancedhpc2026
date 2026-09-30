from numba import cuda
from matplotlib import pyplot
import numpy as np
import time

src = pyplot.imread("big-cat.jpg")
flatSrc = src.reshape(-1, 3)
height = src.shape[0]
width = src.shape[1]
pixelCount = height * width
blockSize = 64

# round up so we don't left any pixels
gridSize = gridSize = (pixelCount + blockSize - 1) // blockSize

@cuda.jit
def grayscale(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    g = np.uint8((src[tidx, 0] + src[tidx, 1] + src[tidx, 2]) / 3)
    dst[tidx, 0] = dst[tidx, 1] = dst[tidx, 2] = g

devSrc = cuda.to_device(flatSrc)
devDst = cuda.device_array((pixelCount, 3), np.uint8)

start = time.time()
grayscale[gridSize, blockSize](devSrc, devDst)
stop = time.time()

hostDst = devDst.copy_to_host().reshape(height, width, 3)

print("Computation time with GPU : ", stop - start)
pyplot.imsave("result_GPU.jpg", hostDst)
