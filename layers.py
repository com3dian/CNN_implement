import numpy as np
import tensorflow  as tf
class convKernel:
    
    def __init__(self, size: tuple):
        self.size = size #kernel size
        
    def initial(self, channel: int):
        self.convKernel, self.bias = [], []
        for i in range(channel):
            self.convKernel.append(np.random.rand(self.size[0], self.size[1]))
            self.bias.append(0)
        # randomly initialize conv kernel
        
    def updateParams(self, learningRate, gradKernel, gradBias):
        self.convKernel -= gradKernel * learningRate # update  kernel weight
        self.bias -= gradBias * learningRate # update kernel bias

def convolution(kernel, data):
    # convolution between single kernel and image
    img_new = []
    
    for i in range(data.shape[0] - kernel.shape[0] + 1):
        line = [] #
        for j in range(data.shape[1] - kernel.shape[1] + 1):
            _a = data[i:(i+kernel.shape[0]), j:(j+kernel.shape[0])]
            line.append(np.sum(np.multiply(_a, kernel)))
        img_new.append(line)
    return np.array(img_new)


def conv4Filters(data, kernels, dataSize:tuple, kernelSize:tuple):
    # data is a 1D-array
    # kernels is 2D array, first dimention is number of kernels, the second is kernel size
    def conv4Kernels(kernel):
        _kernel = np.reshape(kernel, kernelSize)# in our case 3 by 3 
        _data = np.reshape(data, dataSize)# in our case 28 by 28
        
        return convolution(_kernel, _data)
    
    return np.apply_along_axis(conv4Kernels, 1, kernels)


class convLayer:
    
    def __init__(self, numKernel: int, sizeKernel:tuple, inputSize:tuple):
        self.size = sizeKernel
        self.num = numKernel
        self.imageSize = inputSize
        self.kernels = []
        
    def initial(self, channel):
        # color channels
        # 3 if RGB and 1 if balck-white 
        self.kernels = []
        for i in range(self.num):
            k = convKernel(self.size)
            k.initial(channel)
            self.kernels.append(k.convKernel)
            
    def forward(self, data):
        inputSize = (data.shape[-2], data.shape[-1])
        convFilter = np.reshape(np.array(self.kernels), (self.num, self.size[0]*self.size[1]))
        # data should be a 4D array
        xTrain = np.reshape(data, (data.shape[0], data.shape[1], data.shape[-2]*data.shape[-1]))
        convOutput =  np.apply_along_axis(conv4Filters, 2, xTrain, 
                                          convFilter,
                                          self.imageSize, 
                                          self.size)
        return np.reshape(convOutput, (convOutput.shape[0]*convOutput.shape[1]*convOutput.shape[2], 
                                       convOutput.shape[-2], convOutput.shape[-1]))
        
    def backward(delf, output):
        # gradient with respect to bias
        # gradient with respect to weight
        # gradient with respect to input
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

    