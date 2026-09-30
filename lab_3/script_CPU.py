from matplotlib import pyplot
import numpy as np
import time
import sys

src = pyplot.imread(sys.argv[1])
flatSrc = src.reshape(-1, 3)
height = src.shape[0]
width = src.shape[1]

def grayscale(flatImg):
    resultFlatImg = np.zeros((len(flatImg), 3), np.uint8())
    for i in range(len(flatImg)):
        g = np.uint8((int(flatImg[i, 0]) + int(flatImg[i, 1]) + int(flatImg[i, 2])) / 3)
        resultFlatImg[i, 0] = g
        resultFlatImg[i, 1] = g
        resultFlatImg[i, 2] = g

    return resultFlatImg

start = time.time()
result = grayscale(flatSrc)
stop = time.time()

result2D = result.reshape(height, width, 3)

print("CPU : ", stop - start)
pyplot.imsave(sys.argv[2], result2D)
