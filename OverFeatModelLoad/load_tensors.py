import numpy as np
import os
import warnings


def _float_arr_from_file(filepath):
    return np.fromfile(filepath, dtype=np.float32)


def _file_size(filepath):
    return os.stat(filepath).st_size


def load_tensors(data_path, model_nr):
    '''
        data_path - path of the directory containing files
        weights_data_(model_nr)

        model_nr - identifier of the OverFeat model to be loaded.
        The two valid values are: 0 and 1
    '''
    if model_nr not in {0, 1}:
        raise ValueError('Invalid model_nr')

    if model_nr == 0:
        tensor_shapes = [
            (96, 3, 11, 11),
            (96,),
            (256, 96, 5, 5),
            (256,),
            (512, 256, 3, 3),
            (512,),
            (1024, 512, 3, 3),
            (1024,),
            (1024, 1024, 3, 3),
            (1024,),
            (3072, 1024, 6, 6),
            (3072,),
            (4096, 3072, 1, 1),
            (4096,),
            (1000, 4096, 1, 1),
            (1000,)
        ]
    elif model_nr == 1:
        tensor_shapes = [
            (96, 3, 7, 7),
            (96,),
            (256, 96, 7, 7),
            (256,),
            (512, 256, 3, 3),
            (512,),
            (512, 512, 3, 3),
            (512,),
            (1024, 512, 3, 3),
            (1024,),
            (1024, 1024, 3, 3),
            (1024,),
            (4096, 1024, 5, 5),
            (4096,),
            (4096, 4096, 1, 1),
            (4096,),
            (1000, 4096, 1, 1),
            (1000,)
        ]
    else:
        assert(False)

    filepath = data_path + '/net_weight_' + str(model_nr)
    tensor_as_array = _float_arr_from_file(filepath)

    model_file_size = _file_size(filepath)
    if tensor_as_array.nbytes != model_file_size:
        warnings.warn('Loaded array size %ul and file size %ul '
                      ' dont match!' % (tensor_as_array.nbytes,
                                        model_file_size))


    tensors = []
    tensor_start_index = 0

    for tensor_idx in range(len(tensor_shapes)):
        tensor_size = reduce(lambda a, b: a*b,
                             tensor_shapes[tensor_idx])
        tensors.append(np.reshape(tensor_as_array[tensor_start_index:
                                                  tensor_start_index
                                                  + tensor_size],
                                  tensor_shapes[tensor_idx]))
        tensor_start_index = tensor_size

    return tensors
