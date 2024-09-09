import numpy as np
import numpy.typing as npt

def conv_output_shape(
    input_shape: npt.NDArray, kernel_size: npt.NDArray, out_channels: npt.NDArray, 
    dims=2, stride=np.array([1,1]), padding="valid"
):
    output_shape = input_shape.copy()
    output_shape[dims] = out_channels
    if padding == "valid":
        output_shape[:dims] = ((output_shape[:dims] - kernel_size[:dims]) / stride[:dims]) + np.ones(dims)
    elif padding == "same":
        pass
    else:
        panic()
    return output_shape

def max_pool_output_shape(input_shape: npt.NDArray, dims=2, stride=2):
    output_shape = input_shape.copy()
    output_shape[:dims] = output_shape[:dims] / stride
    return output_shape

class ModelInfo:

    def __init__(self, name: str) -> None:
        # Default init values
        self.layers: list[LayerInfo] = []
        self.max_buffer_required = 0
        self.name = name

    def prune(self):
        aux = []
        for l in self.layers:
            if l.type != None:
                aux.append(l)
        self.layers = aux

    def buffer_info(self):
        print(f"Max Buffer {self.max_buffer_required}")
        for l in self.layers:
            l.buffer_info()

    def layer_info(self):
        for l in self.layers:
            l.layer_info()

    def fuse_activations(self):
        for i, l in enumerate(self.layers):
            if l.is_activation:
                self.layers[i-1].activation = l.type
        
        nl = []
        for l in self.layers:
            if not l.is_activation:
                nl.append(l)
        self.layers = nl

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

        # Linear layers info
        self.in_features = None
        self.out_features = None

        # Buffer 
        self.in_buffer_shape = None
        self.out_buffer_shape = None
        self.in_addr = None
        self.out_addr = None

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
        if self.type == "Conv2D":
            print(f"Conv2D({self.in_channels}, {self.out_channels}, {self.kernel_size}, {self.stride}, {self.padding}) [{self.mu_buffer.shape}] {self.activation}")
        elif self.type == "Linear":
            print(f"Linear({self.in_features}, {self.out_features}) [{self.mu_buffer.shape}] {self.activation}")
        elif self.type == "MaxPool2D":
            print(f"MaxPool2D({self.kernel_size}) {self.activation}")
        elif self.type == "ReLU":
            print("ReLU")
        elif self.type == "Softmax":
            print("Softmax")
