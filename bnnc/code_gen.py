from .model_info import ModelInfo, LayerInfo

from math import log2, ceil, sqrt, isinf
import numpy as np
import numpy.typing as npt

import os
bnnc_c_sources_abspath = f"{os.path.dirname(__file__)}/sources_c"

C_DATA_TYPE_RANGES = { 
    "uint8":  [0,        2**8 - 1],
    "int8":   [-(2**7),  2**7 - 1],
    "uint16": [0,        2**16 - 1],
    "int16":  [-(2**15), 2**15 - 1],
    "uint32": [0,        2**32 - 1],
    "int32":  [-(2**31), 2**31 - 1]
}

C_FUNCTION_NAMES = {
    "Conv2D": "bnn_conv2D",
    "Linear": "bnn_linear",
    "MaxPool2D": "layer_max_pooling2D"
}

def saturate_to_data_type(x: npt.NDArray, data_type: str):
    drange = C_DATA_TYPE_RANGES[data_type]
    return np.clip(x, drange[0], drange[1]).astype(int)

# Returns C code string from array
def ndarray_to_c(array, name, fbits, data_type):
    tofixed = 2**fbits

    if len(array.shape) >= 3:
        text = f'// Array {len(array.shape)}D {array.shape}\n'

        if np.prod(array.shape) > (2**32-1):
            print("ERROR, array too big!")

        text += f'{data_type} {name}[{np.prod(array.shape)}] = ' + '{'
        f = array.flatten()
        for v in f:
            aux = saturate_to_data_type(v * tofixed, data_type)
            text += f'{aux + 0}, '
        text = text[:-2]
        text += '};\n'

    elif len(array.shape) == 2:

        lsize = array.shape[1]

        text = f'// Matrix {array.shape[0]} x {lsize}\n'
        text += f'{data_type} {name}[{array.shape[0] * lsize}] = ' + '{'

        for row in array:
            for w in row:
                aux = saturate_to_data_type(w * tofixed, data_type)
                text += f'{aux + 0}, '
            text += '\n'
        text = text[:-3]
        text += '};\n'

    else:
        text = f'// Array {array.shape[0]}\n'
        text += f'{data_type} {name}[{array.shape[0]}] = ' + '{'

        for w in array:
            aux = saturate_to_data_type(w * tofixed, data_type)
            text += f'{aux + 0}, '
        text = text[:-2]
        text += '};\n'

    return text



def layer_id(m: ModelInfo, l: LayerInfo):
    return f"{m.name}_{l.name}"



def model_internal_buffer_id(m: ModelInfo):
    return f"{m.name}_internal_buffer"

def model_internal_buffer_end_id(m: ModelInfo):
    return f"{model_internal_buffer_id(m)}_end"

def create_model_internal_buffers(m: ModelInfo):
    r = ""
    r += f"Data_t {model_internal_buffer_id(m)}[{m.max_buffer_required}];\n"
    r += f"Data_t* const {model_internal_buffer_end_id(m)} = {model_internal_buffer_id(m)} + {m.max_buffer_required};\n\n"
    return r



def buffer_end_ptr(m: ModelInfo, v: int):
    return f"{model_internal_buffer_end_id(m)} {v}"

def l_ptr(m: ModelInfo, v: int):
    if v < 0:
        return buffer_end_ptr(m, v)
    else:
        return model_internal_buffer_id(m)

def layer_inout_ptrs(m: ModelInfo, l: LayerInfo):
    r = ""
    lid = layer_id(m, l)
    
    # Create IN/OUT pointers
    if not l.is_input:
        r += f"Data_t* {lid}_in = {l_ptr(m, l.in_addr)};\n"
    
    out_ptr = l_ptr(m, l.out_addr)
    r += f"Data_t* {lid}_out = {out_ptr};\n"

    return r

def get_layer_input_ptr(m: ModelInfo, l: LayerInfo):
    lid = layer_id(m, l)
    if l.is_input:
        return "data_in"
    else:
        return f"{lid}_in"



def layer_weight_buffers(m: ModelInfo, l: LayerInfo):
    r = ""
    lid = layer_id(m, l)
    r += ndarray_to_c(l.mu_buffer, f"{lid}_mu_buffer", 8, "int32")
    r += ndarray_to_c(l.sigma_buffer, f"{lid}_sigma_buffer", 8, "int32")
    r += ndarray_to_c(l.mu_bias, f"{lid}_mu_bias", 8, "int32")
    #r += ndarray_to_c(l.sigma_bias, f"{lid}_sigma_bias", 8, "int32")
    return r


MODEL_HDR_INCLUDES = """
#include <bnn/layers.h>
"""

MODEL_WEIGHTS_INCLUDES = """
#include <bnn/types.h>
"""

def model_to_c(model: ModelInfo):

    model_buffers_ptrs = ""
    model_weights = ""
    model_fcall = f"Data_t* {model.name}_inference(Data_t* data_in) {{\n"
    model_buffers_ptrs += create_model_internal_buffers(model)

    for l in model.layers:
        lid = layer_id(model, l)
        lfcall = f"\t{C_FUNCTION_NAMES[l.type]}"

        if l.type == "Conv2D":
            
            i, j, k = l.in_buffer_shape
            
            lfcall += f"""_{l.padding}_{l.activation}(
                {get_layer_input_ptr(model, l)},
                {i}, {j}, {k}, {l.out_channels}, {l.kernel_size[0]},
                {lid}_mu_buffer,
                {lid}_sigma_buffer,
                {lid}_mu_bias,
                {lid}_out,
                BNN_SCALE_FACTOR
            );\n"""

            model_weights += layer_weight_buffers(model, l)

        elif l.type == "Linear":

            lfcall += f"""_{l.activation}(
                {lid}_sigma_buffer,
                {lid}_mu_buffer,
                {lid}_mu_bias,
                {get_layer_input_ptr(model, l)},
                {lid}_out,
                BNN_SCALE_FACTOR,
                {l.out_features}, {l.in_features}
            );\n"""

            model_weights += layer_weight_buffers(model, l)

        elif l.type == "MaxPool2D":
            
            i, j, k = l.in_buffer_shape

            lfcall += f"""(
                {get_layer_input_ptr(model, l)},
                {i}, {j}, {k}, {l.kernel_size}, {l.kernel_size},
                {lid}_out 
            );\n"""


        model_buffers_ptrs += layer_inout_ptrs(model, l)
        model_fcall += lfcall
    
    l = model.layers[-1]
    model_fcall += f"\treturn {layer_id(model, l)}_out;\n}}"
    model_buffers_ptrs += f"\nconst size_t {model.name}_num_classes = {l.out_buffer_shape};\n"

    model_hdr = f"{MODEL_HDR_INCLUDES}\n{model_buffers_ptrs}\n{model_fcall}"
    model_weights = f"{MODEL_WEIGHTS_INCLUDES}\n{model_weights}"

    return model_hdr, model_weights
