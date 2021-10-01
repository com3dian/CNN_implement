import numpy as np

def efficient_convolve(image, filters, pad="valid", stride=(1, 1), print_result=False):
    image_y = image.shape[0]
    image_x = image.shape[1]
    channels = image.shape[2]
    number_of_filters = filters.shape[0]
    kernel_y = filters.shape[1]
    kernel_x = filters.shape[2]

    if pad=="same":
        pad_x = kernel_x // 2
        pad_y = kernel_y // 2
    else:
        pad_x = 0
        pad_y = 0
    output_image_x = int((image_x - kernel_x + 2 * pad_x) / stride[1]) + 1
    output_image_y = int((image_y - kernel_y + 2 * pad_y) / stride[0]) + 1
    feature_maps = np.zeros((output_image_y, output_image_x, number_of_filters))
    image_pad = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode='constant')
    
    k_i = np.repeat(np.arange(kernel_y), kernel_x)
    im_i = stride[0] * np.repeat(np.arange(output_image_y), output_image_x)
    k_j = np.tile(np.arange(kernel_x), kernel_y)
    im_j = stride[1] * np.tile(np.arange(output_image_x), output_image_y)
    out_i = k_i.reshape(-1, 1) + im_i.reshape(1, -1)
    out_j = k_j.reshape(-1, 1) + im_j.reshape(1, -1)
    
    vectorized_image = image_pad[out_i, out_j, :].transpose(2, 0, 1).reshape((kernel_x*kernel_y*channels, -1))
    weights = filters.transpose(0, 3, 1, 2).reshape((-1, kernel_x*kernel_y*channels))
    feature_maps = (weights @ vectorized_image).transpose().reshape((output_image_y, output_image_x, number_of_filters))

    if print_result:
        print(feature_maps[:,:,0])
    return feature_maps

def mapData(func, allData, kernels, *args):
    def funcKernels(data):
        if kernels is None:
            return func(data, args)
        else:
            return func(data, kernels, args)
    return list(map(funcKernels, allData))
