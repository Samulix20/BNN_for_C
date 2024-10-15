import numpy as np
import numpy.typing as npt

from math import sqrt, log2, ceil

import os
c_sources_abspath = f"{os.path.dirname(__file__)}/sources_c"

def panic():
    print("PANIC!")
    exit(1)

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

C_INTERNAL_GENERATORS = {
    "Normal": 0,
    "Uniform": 1,
    "Bernoulli": 3
}

def get_datatype(signed: bool, max_val: int) -> str:
    bits = ceil(log2(max_val))
    dt = ""
    if signed:
        bits += 1
    if bits <= 8:
        dt = "int8"
    elif bits <= 16:
        dt = "int16"
    else:
        dt = "int32"
    if not signed:
        dt = "u" + dt
    return dt

def to_fixed(arr: npt.NDArray, fbits: int) -> npt.NDArray:
    return (arr * (2**fbits)).astype(int)

def saturate_to_data_type(x: npt.NDArray, data_type: str) -> npt.NDArray:
    drange = C_DATA_TYPE_RANGES[data_type]
    return np.clip(x, drange[0], drange[1]).astype(int)

def update_data_range(prev_range: tuple[float,float], arr: npt.NDArray) -> tuple[float,float]:
    if prev_range is None:
        return (np.min(arr), np.max(arr))
    else:
        min_v, max_v = prev_range
        return (min(min_v, np.min(arr)), max(max_v, np.max(arr)))

# Returns C code string from array
def ndarray_to_c(array: npt.NDArray, name: str, data_type: str) -> str:

    text = f'// Array {len(array.shape)}D {array.shape}\n'

    if np.prod(array.shape) > (2**32-1):
        print("ERROR, array too big!")

    text += f'{data_type} {name}[{np.prod(array.shape)}] = ' + '{'
    f = array.flatten()
    for v in f:
        aux = saturate_to_data_type(v, data_type)
        text += f'{aux + 0}, '
    text = text[:-2]
    text += '};\n'

    return text

def conv_output_shape(
    input_shape: npt.NDArray, kernel_size: npt.NDArray, out_channels: npt.NDArray, 
    dims=2, stride=np.array([1,1]), padding="valid"
):
    if isinstance(padding, str):
        output_shape = input_shape.copy()
        output_shape[dims] = out_channels
        if padding == "valid":
            output_shape[:dims] = ((output_shape[:dims] - kernel_size[:dims]) / stride[:dims]) + 1
        elif padding == "same":
            pass
    else:
        output_shape[:dims] = np.floor(((output_shape[:dims] + 2 * padding - 1 * (kernel_size - 1) - 1) / stride) + 1)
    return output_shape


def max_pool_output_shape(input_shape: npt.NDArray, dims=2, stride=2):
    output_shape = input_shape.copy()
    output_shape[:dims] = output_shape[:dims] / stride
    return output_shape

def calculate_padding(padding: str|npt.NDArray, input_shape: npt.NDArray, kernel_size: npt.NDArray, stride: npt.NDArray):
    if not isinstance(padding, str):
        return padding

    if padding == "valid":
        return np.array([0,0])
    elif padding == "same":
        output = np.ceil(input_shape[:2]/stride)
        return np.clip(((stride * (output - 1) - input_shape[:2] + (kernel_size - 1) + 1) / 2).astype(int), 0, None)

class ModelInfo:

    def __init__(self, name: str) -> None:
        # Default init values
        self.layers: list[LayerInfo] = []
        self.max_buffer_required = 0
        self.name = name
        # C Info
        self.mc_passes = 100
        self.gen_mode = "Normal"
        self.fixed_bits = 10
        self.data_types = {
            "MU": "int32",
            "SIGMA": "int32",
            "BIAS": "int32",
            "DATA": "int32"
        }

    def print_cinfo(self):
        print(f"Fixed Bits: {self.fixed_bits}")
        print(f"Data Types: {self.data_types}")
        print(f"Generation Mode: {self.gen_mode}")

    def print_buffer_info(self):
        print(f"Max Buffer: {self.max_buffer_required}")
        for l in self.layers:
            l.layer_info()
            #l.buffer_info()

    def internal_buffer_id(self) -> str:
        return f"{self.name}_internal_buffer"

    def internal_buffer_end_id(self) -> str:
        return f"{self.internal_buffer_id()}_end" 
    
    def create_internal_buffers(self) -> str:
        mid = self.internal_buffer_id() 
        meid = self.internal_buffer_end_id()
        r = ""
        r += f"Data_t {mid}[{self.max_buffer_required}];\n"
        r += f"Data_t* const {meid} = {mid} + {self.max_buffer_required};\n\n"
        return r
    
    def buffer_end_ptr(self, v: int):
        return f"{self.internal_buffer_end_id()} {v}"
    
    def buffer_start_or_end_ptr(self, v: int):
        if v < 0:
            return self.buffer_end_ptr(v)
        else:
            return self.internal_buffer_id()

    def prune(self):
        aux = []
        for l in self.layers:
            if l.type != None:
                aux.append(l)
        self.layers = aux

    def fold_layers(self):
        for i, l in enumerate(self.layers):
            if l.type == "BatchNorm2D":
                self.layers[i-1].folded_batch_norm = l
                l.type = None
        self.prune()
        
        for i, l in enumerate(self.layers):
            if l.is_activation:
                self.layers[i-1].activation = l.type
                l.type = None
        self.prune()

        for i, l in enumerate(self.layers):
            if l.type == "ResidualBlock":
                self.layers[i-1].output_to_residual = True
                self.layers[i+1].input_from_residual = True
                l.type = None
        self.prune()

        for i, l in enumerate(self.layers):
            if l.type == "ResidualConv":
                self.layers[i+1].input_from_residual = True
                l.type = None
        self.prune()


    def fuse_residuals(self):
        for i, l in enumerate(self.layers):
            if l.type == "ResidualBlock":
                self.layers[i-1].output_to_residual = True
                self.layers[i+1].input_from_residual = True
        self.prune()

    def calculate_buffers(self, input_shape: npt.NDArray):
        prev_out = None
        prev_shape = None

        for l in self.layers:
            if prev_out is None:

                l.in_buffer_shape = input_shape.copy()
                l.output_shape(input_shape)

                prev_shape = l.out_buffer_shape
                l.in_addr = "Input"
                l.out_addr = 0
                prev_out = 0

            else:

                # Automatic flatten layer before linear
                if l.type == "Linear":
                    prev_shape = np.prod(prev_shape)

                l.in_buffer_shape = prev_shape.copy()
                l.output_shape(prev_shape)

                out = -l.out_buffer_shape.prod() if prev_out == 0 else 0
                l.in_addr = prev_out
                l.out_addr = out

                prev_out = out
                prev_shape = l.out_buffer_shape

        m = np.prod(l.out_buffer_shape)
        for l in self.layers[1:]:
            m = max(m, l.buffer_required())
        self.max_buffer_required = m

    def uniform_weight_transform(self):
        self.gen_mode = "Uniform"

        for l in self.layers:
            if l.type not in ["Conv2D", "Linear"]:
                continue

            b = l.sigma_buffer * sqrt(12.0)
            a = l.mu_buffer - (b / 2)

            l.sigma_buffer = b
            l.mu_buffer = a

    def bernoulli_weight_transform(self):
        self.gen_mode = "Bernoulli"

        for l in self.layers:
            if l.type not in ["Conv2D", "Linear"]:
                continue

            p = l.mu_buffer**2 / (l.mu_buffer**2 + l.sigma_buffer**2)
            q = (l.mu_buffer**2 + l.sigma_buffer**2) / l.mu_buffer

            l.sigma_buffer = q
            l.mu_buffer = p

    def to_fixed(self):
        for l in self.layers:
            if l.type not in ["Conv2D", "Linear"]:
                continue

            l.mu_buffer = to_fixed(l.mu_buffer, self.fixed_bits)
            l.sigma_buffer = to_fixed(l.sigma_buffer, self.fixed_bits)
            l.mu_bias = to_fixed(l.mu_bias, self.fixed_bits)
            l.sigma_bias = to_fixed(l.sigma_bias, self.fixed_bits)

    def find_data_types(self):
        data_range = [None, None, None]
        for l in self.layers:
            if l.type not in ["Conv2D", "Linear"]:
                continue
            # Data range w mu
            data_range[0] = update_data_range(data_range[0], l.mu_buffer)
            # Data range w sigma
            data_range[1] = update_data_range(data_range[1], l.sigma_buffer)
            # Data range b mu
            data_range[2] = update_data_range(data_range[2], l.mu_bias)
            # Data range b sigma (Not yet supported)
        
        for (min_v, max_v), t in zip(data_range, ["MU", "SIGMA", "BIAS"]):
            signed = min_v < 0
            absmax_v = max(abs(min_v), max_v)
            self.data_types[t] = get_datatype(signed, absmax_v)


    def create_lib_config(self) -> str:
        lib_config = ""
        lib_config += "#ifndef BNN_CONFIG_H\n"
        lib_config += "#define BNN_CONFIG_H\n"
        lib_config += f"#define BNN_SIGMA_DT        {self.data_types["SIGMA"]}\n"
        lib_config += f"#define BNN_MU_DT           {self.data_types["MU"]}\n"
        lib_config += f"#define BNN_BIAS_DT         {self.data_types["BIAS"]}\n"
        lib_config += f"#define BNN_DATA_DT         {self.data_types["DATA"]}\n"
        lib_config += f"#define BNN_SCALE_FACTOR {self.fixed_bits}\n"
        lib_config += f"#define BNN_INTERNAL_GEN {C_INTERNAL_GENERATORS[self.gen_mode]}\n"
        lib_config += f"#define BNN_MC_PASSES {self.mc_passes}\n"
        lib_config += "#endif\n"
        return lib_config

    def create_c_code(self) -> tuple[str,str,str]:
        # find data range/type
        self.to_fixed()
        self.find_data_types()

        lib_config = self.create_lib_config()

        model_buffers_ptrs = ""
        model_weights = ""
        model_fcall = f"Data_t* {self.name}_inference(Data_t* data_in) {{\n"
        model_buffers_ptrs += self.create_internal_buffers()

        for l in self.layers:
            lid = l.layer_id(self)
            lfcall = f"\t{C_FUNCTION_NAMES[l.type]}"

            if l.type == "Conv2D":
                
                i, j, k = l.in_buffer_shape

                aux_pad = calculate_padding(l.padding, l.in_buffer_shape, np.array(l.kernel_size), l.stride)
                
                lfcall += f"""_{l.activation}(
                    {l.input_ptr(self)},
                    {i}, {j}, {k}, {l.out_channels}, {l.kernel_size[0]},
                    {aux_pad[0]}, {aux_pad[1]},
                    {l.stride[0]}, {l.stride[1]},
                    {lid}_mu_buffer,
                    {lid}_sigma_buffer,
                    {lid}_mu_bias,
                    {lid}_out,
                    BNN_SCALE_FACTOR
                );\n"""

                model_weights += l.create_weight_buffers(self)

            elif l.type == "Linear":

                lfcall += f"""_{l.activation}(
                    {lid}_sigma_buffer,
                    {lid}_mu_buffer,
                    {lid}_mu_bias,
                    {l.input_ptr(self)},
                    {lid}_out,
                    BNN_SCALE_FACTOR,
                    {l.out_features}, {l.in_features}
                );\n"""

                model_weights += l.create_weight_buffers(self)

            elif l.type == "MaxPool2D":
                
                i, j, k = l.in_buffer_shape

                lfcall += f"""(
                    {l.input_ptr(self)},
                    {i}, {j}, {k}, {l.kernel_size}, {l.kernel_size},
                    {lid}_out 
                );\n"""


            model_buffers_ptrs += l.create_inout_ptrs(self)
            model_fcall += lfcall
        
        l = self.layers[-1]
        model_fcall += f"\treturn {l.layer_id(self)}_out;\n}}"
        model_buffers_ptrs += f"\nconst size_t {self.name}_num_classes = {l.out_buffer_shape};\n"

        model_hdr = f"#include <bnn/layers.h>\n{model_buffers_ptrs}\n{model_fcall}"
        model_weights = f"#include <bnn/types.h>\n{model_weights}"

        return lib_config, model_hdr, model_weights

    def create_c_data(self, data: npt.NDArray) -> str:
        # data [num_data x num_features]

        # find data range/type

        num_data, num_features = data.shape
        c_data = ndarray_to_c(to_fixed(data, self.fixed_bits,), "data_matrix", "int32")

        r = "#include <bnn/types.h>\n"
        r += f"#define NUM_DATA {num_data}\n#define FEATURES_PER_DATA {num_features}"

        return f"{r}\n\n{c_data}"


class LayerInfo:

    def __init__(self):
        # Layer Name
        self.name = None

        # Layer Type
        self.is_input = False
        self.type = None

        # Torch activations are first stored as special layers
        self.is_activation = False
        
        # Activation function String
        self.activation = None

        # Buffers for mu and sigma
        self.mu_buffer = None
        self.sigma_buffer = None
        self.mu_bias = None
        self.sigma_bias = None

        # Info for convolutional type layers
        self.stride = None
        self.kernel_size = None
        self.in_channels = None
        self.out_channels = None
        self.padding = None
        self.folded_batch_norm = None

        # Linear layers info
        self.in_features = None
        self.out_features = None

        # Buffer 
        self.in_buffer_shape = None
        self.out_buffer_shape = None
        self.in_addr = None
        self.out_addr = None

        # Residual I/O
        self.output_to_residual = False
        self.input_from_residual = False

    def layer_id(self, m: ModelInfo):
         return f"{m.name}_{self.name}"

    def input_ptr(self, m: ModelInfo):
        lid = self.layer_id(m)
        if self.is_input:
            return "data_in"
        else:
            return f"{lid}_in"

    def create_inout_ptrs(self, m: ModelInfo):
        r = ""
        lid = self.layer_id(m)
        
        # Create IN/OUT pointers
        if not self.is_input:
            r += f"Data_t* {lid}_in = {m.buffer_start_or_end_ptr(self.in_addr)};\n"
        r += f"Data_t* {lid}_out = {m.buffer_start_or_end_ptr(self.out_addr)};\n"

        return r

    def create_weight_buffers(self, m: ModelInfo):
        r = ""
        lid = self.layer_id(m)
        r += ndarray_to_c(self.mu_buffer, f"{lid}_mu_buffer", m.data_types["MU"])
        r += ndarray_to_c(self.sigma_buffer, f"{lid}_sigma_buffer", m.data_types["SIGMA"])
        r += ndarray_to_c(self.mu_bias, f"{lid}_mu_bias", m.data_types["BIAS"])
        #r += ndarray_to_c(l.sigma_bias, f"{lid}_sigma_bias", 8, "int32")
        return r

    # Output shape size
    def output_shape(self, input_shape: npt.NDArray):
        if self.is_activation:
            self.out_buffer_shape = input_shape
        elif self.type == "MaxPool2D":
            self.out_buffer_shape = max_pool_output_shape(input_shape, dims=2, stride=self.kernel_size)
        elif self.type == "Conv2D":
            self.out_buffer_shape = conv_output_shape(input_shape, self.kernel_size, self.out_channels, dims=2, stride=self.stride, padding=self.padding)
        elif self.type == "Linear":
            self.out_buffer_shape = np.array(self.out_features)
        else:
            panic()
        return self.out_buffer_shape
    
    def buffer_required(self):
        return np.prod(self.in_buffer_shape) + np.prod(self.out_buffer_shape)

    def buffer_info(self):
        print(f"{self.in_buffer_shape} -> {self.type}(in={self.in_addr}, out={self.out_addr}) -> {self.out_buffer_shape} [{self.buffer_required()}]")

    def layer_info(self):
        r = ""

        if self.input_from_residual:
            r += "[Residual ->] "

        if self.type == "Conv2D":
            r += f"{self.name} Conv2D({self.in_channels}, {self.out_channels}, {self.kernel_size}, {self.stride}, {self.padding}) [{self.mu_buffer.shape}] "
        elif self.type == "Linear":
            r += f"{self.name} Linear({self.in_features}, {self.out_features}) [{self.mu_buffer.shape}] "
        elif self.type == "MaxPool2D":
            r += f"{self.name} MaxPool2D({self.kernel_size}) "
        elif self.type == "ResidualAdd":
            r += f"{self.name} ResidualAdd"
        
        else:
            r += f"? {self.name} {self.type} "

        if self.folded_batch_norm is not None:
            r += "[Folded BatchNorm] "

        r += f"[F: {self.activation}] "

        if self.output_to_residual:
            r += "[-> Residual] "

        print(r)
