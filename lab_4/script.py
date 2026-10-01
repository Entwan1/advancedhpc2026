from numba import cuda
from matplotlib import pyplot
import numpy as np
import time

src = pyplot.imread("input/big-cat.jpg")
flatSrc = src.reshape(-1, 3)
height = src.shape[0]
width = src.shape[1]
pixelCount = height * width

devSrc1D = cuda.to_device(flatSrc)
devSrc2D = cuda.to_device(src)

nbIter = 50
blockSizes1D = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
blockSizes2D = [2, 4, 8, 16, 32]
results1D = []
results2D = []

@cuda.jit
def grayscale2D(src, dst):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if y > src.shape[0] or x > src.shape[1]:
        return
    g = np.uint8((src[y, x, 0] + src[y, x, 1] + src[y, x, 2]) / 3)
    dst[y, x, 0] = dst[y, x, 1] = dst[y, x, 2] = g

@cuda.jit
def grayscale1D(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tidx > src.shape[0]:
        return
    g = np.uint8((src[tidx, 0] + src[tidx, 1] + src[tidx, 2]) / 3)
    dst[tidx, 0] = dst[tidx, 1] = dst[tidx, 2] = g

for value in blockSizes1D:
    gridSize1D = (pixelCount + value - 1) // value
    total = 0
    devDst1D = cuda.to_device(np.zeros((pixelCount, 3), np.uint8))
    grayscale1D[gridSize1D, value](devSrc1D, devDst1D)
    for i in range(nbIter):
        start = time.time()
        grayscale1D[gridSize1D, value](devSrc1D, devDst1D)
        cuda.synchronize()
        stop = time.time()
        total += stop - start

    results1D.append(total / nbIter)

for value in blockSizes2D:
    blockSize2D = (value, value)
    gridSize2D = ((width + value - 1) // value, (height + value -1) // value)
    total = 0
    devDst2D = cuda.to_device(np.zeros((height, width, 3), np.uint8))
    grayscale2D[gridSize2D, blockSize2D](devSrc2D, devDst2D)
    for i in range(nbIter):
        start = time.time()
        grayscale2D[gridSize2D, blockSize2D](devSrc2D, devDst2D)
        cuda.synchronize()
        stop = time.time()
        total += stop - start

    results2D.append(total / nbIter)
    
pyplot.figure()
pyplot.bar([str(value) for value in blockSizes2D], results2D)
pyplot.title('Block size benchmark 2D')
pyplot.ylabel('temps(s)')
pyplot.savefig("output/chart2D")

pyplot.figure()
pyplot.bar([str(value) for value in blockSizes1D], results1D)
pyplot.title('Block size benchmark 1D')
pyplot.ylabel('temps(s)')
pyplot.savefig("output/chart1D")

pyplot.figure()
pyplot.bar(['2D', '1D'], [min(results2D), min(results1D)])
pyplot.title('Execution time comparison between 2D and 1D')
pyplot.ylabel('temps(s)')
pyplot.savefig("output/chartComp1Dvs2D")

hostDst1D = devDst1D.copy_to_host().reshape(height, width, 3)
hostDst2D = devDst2D.copy_to_host()

pyplot.imsave("output/result_1D.jpg", hostDst1D)
pyplot.imsave("output/result_2D.jpg", hostDst2D)
