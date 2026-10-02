from numba import cuda
from matplotlib import pyplot
import numpy as np
import time

src = pyplot.imread("input/big-cat.jpg")
flatSrc = src.reshape(-1, 3)
height = src.shape[0]
width = src.shape[1]
pixelCount = height * width
nbIter = 50
blockSizes = [2, 4, 8, 16, 32]
grid = np.array([
    [0, 0, 1, 2, 1, 0, 0],
    [0, 3, 13, 22, 13, 3, 0],
    [1, 13, 59, 97, 59, 13, 1],
    [2, 22, 97, 159, 97, 22, 2],
    [1, 13, 59, 97, 59, 13, 1],
    [0, 3, 13, 22, 13, 3, 0],
    [0, 0, 1, 2, 1, 0, 0]
])

devSrc = cuda.to_device(src)
devDst = cuda.to_device(np.zeros((height, width, 3), np.uint8))

devSrcShared = cuda.to_device(src)
devDstShared = cuda.to_device(np.zeros((height, width, 3), np.uint8))

devGrid = cuda.to_device(grid)

GRID_DIM = 4

def gaussianBlurCPU(src, gaussianGrid):
    height = len(src)
    width = len(src[0])
    result = np.zeros((height, width, 3), np.uint8)
    for i in range(3, height - 3):
        for j in range(3, height - 3):
            tot_r = tot_g = tot_b = 0
            for v in range(-3, 4):
                for w in range(-3, 4):
                    tot_r += gaussianGrid[v + 3][w + 3]*int(src[i + v][j + w][0])
                    tot_g += gaussianGrid[v + 3][w + 3]*int(src[i + v][j + w][1])
                    tot_b += gaussianGrid[v + 3][w + 3]*int(src[i + v][j + w][2])
        
            result[i][j][0] = tot_r/1003
            result[i][j][1] = tot_g/1003
            result[i][j][2] = tot_b/1003

    return result

@cuda.jit
def gaussianBlur(src, dst, grid):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if y > src.shape[0] or x > src.shape[1]:
        return
    tot_r = tot_g = tot_b = 0
    for v in range(-3, 4):
        for w in range(-3, 4):
            tot_r += grid[v + 3][w + 3]*int(src[y + v][x + w][0])
            tot_g += grid[v + 3][w + 3]*int(src[y + v][x + w][1])
            tot_b += grid[v + 3][w + 3]*int(src[y + v][x + w][2])

    dst[y, x, 0] = tot_r/1003
    dst[y, x, 1] = tot_g/1003
    dst[y, x, 2] = tot_b/1003

@cuda.jit
def gaussianBlurShared(src, dst, grid):
    xBlock = cuda.blockIdx.x * cuda.blockDim.x
    yBlock = cuda.blockIdx.y * cuda.blockDim.y
    xInBlock = cuda.threadIdx.x
    yInBlock = cuda.threadIdx.y
    x = xInBlock + xBlock
    y = yInBlock + yBlock
    shared_r = cuda.shared.array((16, 16), np.uint8)
    shared_g = cuda.shared.array((16, 16), np.uint8)
    shared_b = cuda.shared.array((16, 16), np.uint8)
    shared_r[yInBlock, xInBlock] = src[y, x, 0]
    shared_g[yInBlock, xInBlock] = src[y, x, 1]
    shared_b[yInBlock, xInBlock] = src[y, x, 2]
    cuda.syncthreads()
    if y > src.shape[0] or x > src.shape[1]:
        return
    tot_r = tot_g = tot_b = 0
    for v in range(-3, 4):
        for w in range(-3, 4):
            tot_r += grid[v + 3][w + 3]*int(shared_r[yInBlock + v][xInBlock + w])
            tot_g += grid[v + 3][w + 3]*int(shared_g[yInBlock + v][xInBlock + w])
            tot_b += grid[v + 3][w + 3]*int(shared_b[yInBlock + v][xInBlock + w])
    
    dst[y, x, 0] = tot_r/1003
    dst[y, x, 1] = tot_g/1003
    dst[y, x, 2] = tot_b/1003


blockSize = (GRID_DIM, GRID_DIM)
gridSize = ((width + GRID_DIM - 1) // GRID_DIM, (height + GRID_DIM -1) // GRID_DIM)
total = 0
devDstShared = cuda.to_device(np.zeros((height, width, 3), np.uint8))
gaussianBlurShared[gridSize, blockSize](devSrcShared, devDstShared, devGrid)
for i in range(nbIter):
    start = time.time()
    gaussianBlurShared[gridSize, blockSize](devSrcShared, devDstShared, devGrid)
    cuda.synchronize()
    stop = time.time()
    total += stop - start

print("Shared:", total / nbIter)

blockSize = (GRID_DIM, GRID_DIM)
gridSize = ((width + GRID_DIM - 1) // GRID_DIM, (height + GRID_DIM -1) // GRID_DIM)
total = 0
devDst = cuda.to_device(np.zeros((height, width, 3), np.uint8))
gaussianBlur[gridSize, blockSize](devSrc, devDst, devGrid)
for i in range(nbIter):
    start = time.time()
    gaussianBlur[gridSize, blockSize](devSrc, devDst, devGrid)
    cuda.synchronize()
    stop = time.time()
    total += stop - start

print("Not shared:", total / nbIter)

# Writen by hand because the block size cannot be change dinamically
sharedResults = [0.02853285312652588, 0.014578189849853516, 0.01511357307434082,  0.021281423568725585]
results = [0.04518040180206299,  0.02354569435119629, 0.020146121978759767, 0.024509572982788087]

pyplot.figure()
pyplot.bar(["4", "8", "16", "32"], results)
pyplot.title('Block size benchmark without shared memory')
pyplot.ylabel('time (s)')
pyplot.savefig("output/chart")

pyplot.figure()
pyplot.bar(["4", "8", "16", "32"], sharedResults)
pyplot.title('Block size benchmark with shared memory')
pyplot.ylabel('time (s)')
pyplot.savefig("output/chartShared")

pyplot.figure()
pyplot.bar(['Not shared', 'Shared'], [min(results), min(sharedResults)])
pyplot.title('Execution time comparison between 2D and 1D')
pyplot.ylabel('time (s)')
pyplot.savefig("output/chartComp")

hostDstShared = devDstShared.copy_to_host()
hostDst = devDst.copy_to_host()

pyplot.imsave("output/result_shared.jpg", hostDstShared)
pyplot.imsave("output/result_not_shared.jpg", hostDst)