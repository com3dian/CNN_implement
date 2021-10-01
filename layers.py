import numpy as np
import tensorflow  as tf
from funcs import efficient_convolve, mapData
import numpy as np

class convKernel:
    
    def __init__(self, size: tuple):
        self.size = size #kernel size
        
    def initial(self):
        self.convKernel = np.random.rand(self.size[0], self.size[1], self.size[2])
        self.bias = 0
        # randomly initialize conv kernel
        
class convLayer:
    
    def __init__(self, numKernel: int, sizeKernel:tuple, inputSize:tuple):
        self.size = sizeKernel
        self.num = numKernel
        self.imageSize = inputSize
        self.kernels = []
        
    def initial(self):
        # color channels
        # 3 if RGB and 1 if balck-white 
        self.kernels = []
        for i in range(self.num):
            k = convKernel(self.size)
            k.initial()
            self.kernels.append(k.convKernel)
    def forward(self, data):
        return mapData(efficient_convolve, data, np.array(self.kernels), "valid", (1, 1), False)
    def backward(self, grad):
        pass
        
        

        
class reluActivators(object):
    def relu(self, data):
        return np.where(data>0, data, 0)
        
    def backward(self,output):
        return 1 if output > 0 else 0 # gradient with respect to input
    
class maxpoolingLayer(object):
    # without stride 
    # always zero padding
    def __init__(self, size):
        self.poolSize = size
        
    def maxpooling(self, data):
        # data is 4D array
        self.shape = data.shape #data shape
        def maxFind(image):
            # name is shit
            image = np.reshape(image, (self.shape[-2], self.shape[-1]))
            self.output = np.zeros(image.shape)# initialize
            for m in range(image.shape[-2] - self.poolSize[0]):
                for n in range(image.shape[-1] - self.poolSize[1]):
                    imageSquare = image[m:(m+self.poolSize[0]), n:(n+self.poolSize[1])] # somehow stupid
                    self.output[m, n] = np.max(imageSquare)
            return self.output
        
        data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2]*data.shape[3]))
        return np.reshape(np.apply_along_axis(maxFind, 1, data), self.shape)
    
    def backward(self, output):
        self.shape = output.shape #data shape
        def maxBP(image):
            # name is shit
            image = np.reshape(image, (self.shape[-2], self.shape[-1]))
            grad = np.zeros(image.shape) # save for backward
            for m in range(image.shape[-2] - self.poolSize[0]):
                for n in range(image.shape[-1] - self.poolSize[1]):
                    imageSquare = image[m:(m+self.poolSize[0]), n:(n+self.poolSize[1])] # somehow stupid
                    grad[np.where(imageSquare == np.max(imageSquare) )[0]+m, 
                              np.where(imageSquare == np.max(imageSquare) )[1]+n] = 1
            return grad
        
        data = np.reshape(output, (output.shape[0]*output.shape[1], output.shape[2]*output.shape[3]))
        return np.reshape(np.apply_along_axis(maxBP, 1, data), self.shape)


class normalizationLayer:
    def normalize(self, data):
        # data is 4D array
        self.dataSize = data.shape
        imageSize = (data.shape[-2], data.shape[-1]) # shold be improved
        data = np.reshape(data, (data.shape[0], data.shape[-2]*data.shape[-1]))
        # reshape to fit apply_along_axis
        sqrt = lambda x: np.sqrt(np.sum((x- np.mean(x))**2)/(imageSize[0]*imageSize[1]))
        def norm(image):
            normalized_image = (image - np.mean(image)) / sqrt(image)
            return normalized_image
        
        self.standardDeviation = np.apply_along_axis(sqrt, 1, data) # save the sqrt for backward
        return np.reshape(np.apply_along_axis(norm, 1, data), self.dataSize)
    
    def backward(self, output):
        # d(output)/d(normalization)
        return ((1/self.standardDeviation) * np.ones(self.dataSize).T).T # gradient with respect to input

    